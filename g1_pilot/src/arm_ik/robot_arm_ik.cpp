#include "arm_ik/robot_arm_ik.h"

#include <pinocchio/autodiff/casadi.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/collision/collision.hpp>

#include <memory>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <cmath>

// ============================================================================
// HumanoidIK Implementation
// ============================================================================

namespace HumanoidPilot {

namespace {

casadi::GenericType yaml_to_generic(const YAML::Node& node, const std::string& key) {
    if (node.IsMap()) {
        casadi::Dict dict;
        for (const auto& kv : node) {
            const std::string name = kv.first.as<std::string>();
            dict[name] = yaml_to_generic(kv.second, key + "." + name);
        }
        return dict;
    }
    if (node.IsSequence()) {
        // Lists of strings only, e.g. jit_options.flags.
        std::vector<std::string> list;
        for (const auto& item : node) {
            list.push_back(item.as<std::string>());
        }
        return list;
    }
    if (!node.IsScalar()) {
        throw std::runtime_error("ik_solver option " + key + " must be a scalar, list or map");
    }
    const std::string text = node.Scalar();
    // Quoted scalars ("yes", "exact") stay strings.
    if (node.Tag() != "!") {
        if (text == "true") return true;
        if (text == "false") return false;
        try {
            std::size_t used = 0;
            const long long value = std::stoll(text, &used);
            if (used == text.size()) return static_cast<casadi_int>(value);
        } catch (const std::exception&) {}
        try {
            std::size_t used = 0;
            const double value = std::stod(text, &used);
            if (used == text.size()) return value;
        } catch (const std::exception&) {}
    }
    return text;
}

} // namespace

IKSolverOptions parse_ik_solver_options(const YAML::Node& section) {
    IKSolverOptions result;
    if (!section) {
        return result;
    }
    if (section["plugin"]) {
        result.plugin = section["plugin"].as<std::string>();
    }
    if (section["hessian"]) {
        const std::string hessian = section["hessian"].as<std::string>();
        if (hessian != "exact" && hessian != "gauss_newton") {
            throw std::runtime_error(
                "ik_solver.hessian must be exact or gauss_newton, not " + hessian);
        }
        result.gauss_newton_hessian = hessian == "gauss_newton";
    }
    if (const YAML::Node options = section["options"]) {
        if (!options.IsMap()) {
            throw std::runtime_error("ik_solver.options must be a map");
        }
        // A configured solver replaces the default IPOPT options entirely.
        result.options.clear();
        for (const auto& kv : options) {
            const std::string name = kv.first.as<std::string>();
            result.options[name] = yaml_to_generic(kv.second, name);
        }
    }
    return result;
}

HumanoidIK::HumanoidIK(
    pinocchio::Model& model,
    pinocchio::GeometryModel& geom_model,
    const IKSolverOptions& solver_options
) : model_(model), geom_model_(geom_model)
{
    // Get required joint IDs    
    L_hand_id_ = model_.getFrameId("L_ee");
    R_hand_id_ = model_.getFrameId("R_ee");
    // oMcamera = model_.getFrameId("d435_link", pinocchio::BODY);
    // oMLidar = model_.getFrameId("mid360_link", pinocchio::BODY);

    // Setup Collision
    geom_model_.addAllCollisionPairs();    
    filter_adjacent_collision_pairs();

    // Setup IK problem
    setup_optimization(solver_options);

    // Initialize data structures
    data_ = pinocchio::Data(model_);
    geom_data_ = pinocchio::GeometryData(geom_model_);
    init_data_ = Eigen::VectorXd::Zero(model_.nq);

    nq_ = model_.nq;
    nv_ = model_.nv;

}

bool HumanoidIK::check_collision(const Eigen::VectorXd& q) {
    return pinocchio::computeCollisions(
        model_, data_, geom_model_, geom_data_, q, true
    );
}

void HumanoidIK::filter_adjacent_collision_pairs() {
    // Get kinematic adjacency from the model
    std::set<std::pair<int, int>> adjacent_pairs;
    for (int i = 1; i < model_.njoints; ++i) {
        adjacent_pairs.insert({model_.parents[i], i});
        adjacent_pairs.insert({i, model_.parents[i]});
    }
    
    // Filter out neighboring links
    std::vector<pinocchio::CollisionPair> filtered_pairs;
    for (const auto& cp : geom_model_.collisionPairs) {
        int link1 = geom_model_.geometryObjects[cp.first].parentJoint;
        int link2 = geom_model_.geometryObjects[cp.second].parentJoint;
        
        if (adjacent_pairs.find({link1, link2}) == adjacent_pairs.end()) {
            filtered_pairs.push_back(cp);
        }
    }
    
    geom_model_.collisionPairs = filtered_pairs;
}

void HumanoidIK::setup_optimization(const IKSolverOptions& solver_options) {

    pinocchio::ModelTpl<casadi::SX> cmodel(model_.cast<casadi::SX>());
    pinocchio::DataTpl<casadi::SX> cdata(cmodel);

    casadi::SX cq = casadi::SX::sym("q", cmodel.nq);
    casadi::SX cq_last = casadi::SX::sym("q", cmodel.nq);
    casadi::SX cTf_l = casadi::SX::sym("tf_l", 4, 4);
    casadi::SX cTf_r = casadi::SX::sym("tf_r", 4, 4);

    Eigen::Matrix<casadi::SX, Eigen::Dynamic, 1> cq_eigen(cmodel.nq);
    Eigen::Matrix<casadi::SX,4,4>  Target_L_eig;
    Eigen::Matrix<casadi::SX,4,4>  Target_R_eig;

    pinocchio::casadi::copy(cq, cq_eigen);
    pinocchio::casadi::copy(cTf_l, Target_L_eig);
    pinocchio::casadi::copy(cTf_r, Target_R_eig);

    auto L_hand_id = cmodel.getFrameId("L_ee");
    auto R_hand_id = cmodel.getFrameId("R_ee");

    pinocchio::framesForwardKinematics(cmodel, cdata, cq_eigen);
    pinocchio::SE3Tpl<casadi::SX> T_L = cdata.oMf[L_hand_id];
    pinocchio::SE3Tpl<casadi::SX> T_R = cdata.oMf[R_hand_id];

    pinocchio::SE3Tpl<casadi::SX>  Target_L(Target_L_eig);
    pinocchio::SE3Tpl<casadi::SX>  Target_R(Target_R_eig);

    casadi::SX Err_L(casadi::Sparsity::dense(6,1));
    casadi::SX Err_R(casadi::Sparsity::dense(6,1));

    pinocchio::casadi::copy(
        pinocchio::log6(T_L.actInv(Target_L)).toVector(),
        Err_L
    );

    pinocchio::casadi::copy(
        pinocchio::log6(T_R.actInv(Target_R)).toVector(),
        Err_R
    );

    casadi::SX left_trans_err = Err_L(casadi::Slice(0,3),0);
    casadi::SX right_trans_err = Err_R(casadi::Slice(0,3),0);
    casadi::SX left_rot_err = Err_L(casadi::Slice(3,6),0);
    casadi::SX right_rot_err = Err_R(casadi::Slice(3,6),0);

    casadi::SX both_arm_cost = (
        50.0 * (
            casadi::SX::sumsqr(left_trans_err) +
            casadi::SX::sumsqr(right_trans_err)
        ) + 
        3.0 * (
            casadi::SX::sumsqr(left_rot_err) +
            casadi::SX::sumsqr(right_rot_err)
        ) + 
        0.005 * casadi::SX::sumsqr(cq) +
        0.1 * casadi::SX::sumsqr(cq - cq_last)
    );

    casadi::SX left_arm_cost = (
        50.0 * casadi::SX::sumsqr(left_trans_err) + 
        3.0 * casadi::SX::sumsqr(left_rot_err) + 
        0.005 * casadi::SX::sumsqr(cq) +
        0.1 * casadi::SX::sumsqr(cq - cq_last)
    );

    casadi::SX right_arm_cost = (
        50.0 * casadi::SX::sumsqr(right_trans_err) + 
        3.0 * casadi::SX::sumsqr(right_rot_err) + 
        0.005 * casadi::SX::sumsqr(cq) +
        0.1 * casadi::SX::sumsqr(cq - cq_last)
    );

    // Parameters: the previous solution, then each target column-major (as
    // eigen_to_casadi lays it out).
    const casadi::SX vec_tf_l = casadi::SX::vec(cTf_l);
    const casadi::SX vec_tf_r = casadi::SX::vec(cTf_r);

    lower_limits_ = eigen_to_casadi(model_.lowerPositionLimit);
    upper_limits_ = eigen_to_casadi(model_.upperPositionLimit);

    // Gauss-Newton Hessian: 2 J^T J of the weighted pose residual plus the
    // (exact) Hessian of the two quadratic regularization terms. Positive
    // definite and much cheaper to evaluate than the exact Hessian of log6.
    auto gauss_newton_hessian = [&](
        const std::string& name, const casadi::SX& p,
        const casadi::SX& pose_residual
    ) {
        const casadi::SX lam_f = casadi::SX::sym("lam_f");
        const casadi::SX lam_g = casadi::SX::sym("lam_g", 0, 1);
        const casadi::SX J = casadi::SX::jacobian(pose_residual, cq);
        const casadi::SX H = 2.0 * casadi::SX::mtimes(J.T(), J) +
            2.0 * (0.005 + 0.1) * casadi::SX::eye(cmodel.nq);
        // IPOPT takes the upper triangle, sqpmethod the full matrix.
        const bool triu = solver_options.plugin == "ipopt";
        return casadi::Function(
            name + "_hess_lag",
            {cq, p, lam_f, lam_g},
            {triu ? casadi::SX::triu(lam_f * H) : lam_f * H},
            {"x", "p", "lam_f", "lam_g"},
            {triu ? "triu_hess_gamma_x_x" : "hess_gamma_x_x"}
        );
    };
    auto make_solver = [&](
        const std::string& name, const casadi::SX& p, const casadi::SX& cost,
        const casadi::SX& pose_residual
    ) {
        casadi::Dict opts = solver_options.options;
        if (solver_options.gauss_newton_hessian) {
            opts["hess_lag"] = gauss_newton_hessian(name, p, pose_residual);
        }
        return casadi::nlpsol(
            name, solver_options.plugin,
            casadi::SXDict{{"x", cq}, {"p", p}, {"f", cost}},
            opts
        );
    };

    const casadi::SX left_residual = casadi::SX::vertcat({
        std::sqrt(50.0) * left_trans_err, std::sqrt(3.0) * left_rot_err
    });
    const casadi::SX right_residual = casadi::SX::vertcat({
        std::sqrt(50.0) * right_trans_err, std::sqrt(3.0) * right_rot_err
    });

    both_solver_ = make_solver(
        "ik_both", casadi::SX::vertcat({cq_last, vec_tf_l, vec_tf_r}),
        both_arm_cost, casadi::SX::vertcat({left_residual, right_residual})
    );
    left_solver_ = make_solver(
        "ik_left", casadi::SX::vertcat({cq_last, vec_tf_l}),
        left_arm_cost, left_residual
    );
    right_solver_ = make_solver(
        "ik_right", casadi::SX::vertcat({cq_last, vec_tf_r}),
        right_arm_cost, right_residual
    );

    both_lam_x_ = casadi::DM::zeros(model_.nq, 1);
    left_lam_x_ = casadi::DM::zeros(model_.nq, 1);
    right_lam_x_ = casadi::DM::zeros(model_.nq, 1);

}

Eigen::VectorXd HumanoidIK::solve(
    casadi::Function& solver,
    casadi::DM& lam_x,
    const casadi::DM& params
)
{
    casadi::DMDict result = solver(casadi::DMDict{
        {"x0", eigen_to_casadi(init_data_)},
        {"p", params},
        {"lbx", lower_limits_},
        {"ubx", upper_limits_},
        {"lam_x0", lam_x}
    });

    const casadi::Dict stats = solver.stats();
    const auto success = stats.find("success");
    if (success == stats.end() || !success->second.to_bool()) {
        const auto status = stats.find("return_status");
        std::cerr << "ERROR in convergence: "
                  << (status != stats.end() ? status->second.to_string() : "unknown")
                  << std::endl;
    }

    lam_x = result.at("lam_x");
    const std::vector<double> sol_q_vec =
        static_cast<std::vector<double>>(result.at("x"));
    return Eigen::Map<const Eigen::VectorXd>(sol_q_vec.data(), model_.nq);
}

void HumanoidIK::reset() {
    init_data_ = Eigen::VectorXd::Zero(model_.nq);
}

std::pair<double, double> HumanoidIK::compute_error(Eigen::Matrix4d A, Eigen::Matrix4d B){
    return {
        (A.block<3,1>(0,3) - B.block<3,1>(0,3)).norm(),
        std::acos(std::clamp(
            0.5 * ((
                A.block<3, 3>(0, 0) * B.block<3, 3>(0, 0).transpose()
            ).trace() - 1),
            -1.0,1.0
        ))
    };
}

JointState HumanoidIK::solve_ik(
    const Eigen::Matrix4d& left_wrist,
    const Eigen::Matrix4d& right_wrist,
    const Eigen::VectorXd* current_lr_arm_motor_q,
    const Eigen::VectorXd* current_lr_arm_motor_dq,
    const Eigen::VectorXd* EE_efrc_L,
    const Eigen::VectorXd* EE_efrc_R,
    double* l_pos_err,
    double* l_rot_err,
    double* r_pos_err,
    double* r_rot_err,
    bool* collision
)
{
    // Update initial guess
    if (current_lr_arm_motor_q != nullptr) {
        init_data_ = *current_lr_arm_motor_q;
    }

    // Solve from, and regularize toward, the initial guess
    sol_q = solve(both_solver_, both_lam_x_, casadi::DM::vertcat({
        eigen_to_casadi(init_data_),
        casadi::DM::vec(eigen_to_casadi(left_wrist)),
        casadi::DM::vec(eigen_to_casadi(right_wrist))
    }));

    pinocchio::framesForwardKinematics(model_, data_, sol_q);

    if (l_pos_err != nullptr || l_rot_err != nullptr) {
        auto [pe, re] = compute_error(
            data_.oMf[L_hand_id_].toHomogeneousMatrix(), left_wrist
        );
        if (l_pos_err) *l_pos_err = pe;
        if (l_rot_err) *l_rot_err = re;
    }
    if (r_pos_err != nullptr || r_rot_err != nullptr) {
        auto [pe, re] = compute_error(
            data_.oMf[R_hand_id_].toHomogeneousMatrix(), right_wrist
        );
        if (r_pos_err) *r_pos_err = pe;
        if (r_rot_err) *r_rot_err = re;
    }

    if (collision != nullptr) {
        *collision = pinocchio::computeCollisions(
            model_, data_, geom_model_, geom_data_, sol_q, true
        );
    }

    // Compute velocity
    sol_v = Eigen::VectorXd::Zero(model_.nv);
    
    // Compute feedforward torques using RNEA
    sol_t = pinocchio::rnea(
        model_, data_, sol_q, sol_v,
        Eigen::VectorXd::Zero(model_.nv)
    );

    generalize_ext_wrenches(EE_efrc_L, EE_efrc_R);

    JointState result = vectors_to_jointstate(
        sol_q, sol_v, sol_t, model_
    );
    init_data_ = sol_q;
    
    return result;

}

JointState HumanoidIK::solve_ik(
    const Eigen::Matrix4d& wrist,
    const bool left,
    const Eigen::VectorXd* current_lr_arm_motor_q,
    const Eigen::VectorXd* current_lr_arm_motor_dq,
    const Eigen::VectorXd* EE_efrc,
    double* pos_err,
    double* rot_err,
    bool* collision
)
{
    // Update initial guess
    if (current_lr_arm_motor_q != nullptr) {
        init_data_ = *current_lr_arm_motor_q;
    }

    // Solve from, and regularize toward, the initial guess
    const casadi::DM params = casadi::DM::vertcat({
        eigen_to_casadi(init_data_),
        casadi::DM::vec(eigen_to_casadi(wrist))
    });
    sol_q = left
        ? solve(left_solver_, left_lam_x_, params)
        : solve(right_solver_, right_lam_x_, params);

    pinocchio::framesForwardKinematics(model_, data_, sol_q);

    if (pos_err != nullptr || rot_err != nullptr) {
        auto frame_id = left ? L_hand_id_ : R_hand_id_;
        auto [pe, re] = compute_error(
            data_.oMf[frame_id].toHomogeneousMatrix(), wrist
        );
        if (pos_err) *pos_err = pe;
        if (rot_err) *rot_err = re;
    }

    if (collision != nullptr) {
        *collision = pinocchio::computeCollisions(
            model_, data_, geom_model_, geom_data_, sol_q, true
        );
    }

    // Compute velocity
    sol_v = Eigen::VectorXd::Zero(model_.nv);
    
    // Compute feedforward torques using RNEA
    sol_t = pinocchio::rnea(
        model_, data_, sol_q, sol_v,
        Eigen::VectorXd::Zero(model_.nv)
    );

    if (left) generalize_ext_wrenches(EE_efrc, nullptr);
    else generalize_ext_wrenches(nullptr, EE_efrc);

    JointState result = vectors_to_jointstate(
        sol_q, sol_v, sol_t, model_
    );
    init_data_ = sol_q;
    
    return result;

}

void HumanoidIK::generalize_ext_wrenches(
    const Eigen::VectorXd* EE_efrc_L,
    const Eigen::VectorXd* EE_efrc_R
){
    if (EE_efrc_L == nullptr && EE_efrc_R == nullptr)
        return;

    if (EE_efrc_L != nullptr){

        J_L = pinocchio::Data::Matrix6x::Zero(6, nv_);

        pinocchio::computeFrameJacobian(
            model_, data_, sol_q, 
            L_hand_id_, pinocchio::LOCAL_WORLD_ALIGNED, 
            J_L
        );

        sol_t = J_L.transpose() * (*EE_efrc_L);
    }

    if (EE_efrc_R != nullptr){

        J_R = pinocchio::Data::Matrix6x::Zero(6, nv_);

        pinocchio::computeFrameJacobian(
            model_, data_, sol_q, 
            R_hand_id_, pinocchio::LOCAL_WORLD_ALIGNED, 
            J_R
        );

        sol_t = J_R.transpose() * (*EE_efrc_R);
    }

    

}

} // HumanoidPilot namespace
