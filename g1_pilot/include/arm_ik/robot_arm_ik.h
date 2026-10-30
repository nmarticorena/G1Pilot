#ifndef ROBOT_ARM_IK_H
#define ROBOT_ARM_IK_H

#include "utils.h"
#include "../base/interfaces.h"

#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/geometry.hpp>

#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include <vector>
#include <string>


namespace HumanoidPilot {

// casadi nlpsol plugin ("ipopt", "sqpmethod", ...) and its options, e.g.
// "ipopt.max_iter" or "qpsol_options". The defaults reproduce the original
// hard-coded IPOPT setup.
struct IKSolverOptions {
    std::string plugin = "ipopt";
    casadi::Dict options = {
        {"print_time", false},
        {"ipopt.sb", "yes"},
        {"ipopt.print_level", 0},
        {"ipopt.max_iter", 50},
        {"ipopt.tol", 1e-6},
        {"ipopt.acceptable_tol", 5e-4},
        {"ipopt.acceptable_iter", 5},
        {"ipopt.warm_start_init_point", "yes"},
        {"ipopt.derivative_test", "none"},
        {"ipopt.jacobian_approximation", "exact"},
    };
    // Replace the exact Hessian of the cost with its Gauss-Newton
    // approximation (ik_solver.hessian: gauss_newton).
    bool gauss_newton_hessian = false;
};

// Reads an ik_solver section ({plugin, hessian, options}); a missing
// section or key keeps the defaults above. Option values keep their YAML
// types: integers, floats, true/false, strings and nested maps.
IKSolverOptions parse_ik_solver_options(const YAML::Node& section);

class HumanoidIK {
public:
    HumanoidIK(
        pinocchio::Model& model,
        pinocchio::GeometryModel& geom_model,
        const IKSolverOptions& solver_options = IKSolverOptions()
    );

    virtual JointState solve_ik(
        const Eigen::Matrix4d& left_wrist,
        const Eigen::Matrix4d& right_wrist,
        const Eigen::VectorXd* current_lr_arm_motor_q = nullptr,
        const Eigen::VectorXd* current_lr_arm_motor_dq = nullptr,
        const Eigen::VectorXd* EE_efrc_L = nullptr,
        const Eigen::VectorXd* EE_efrc_R = nullptr,
        double* l_pos_err = nullptr,
        double* l_rot_err = nullptr,
        double* r_pos_err = nullptr,
        double* r_rot_err = nullptr,
        bool* collision = nullptr
    );

    virtual JointState solve_ik(
        const Eigen::Matrix4d& wrist,
        const bool left = false,
        const Eigen::VectorXd* current_lr_arm_motor_q = nullptr,
        const Eigen::VectorXd* current_lr_arm_motor_dq = nullptr,
        const Eigen::VectorXd* EE_efrc = nullptr,
        double* pos_err = nullptr,
        double* rot_err = nullptr,
        bool* collision = nullptr
    );

    bool check_collision(const Eigen::VectorXd& q);

    void reset();

protected:
    void setup_optimization(const IKSolverOptions& solver_options);
    // Runs one of the solvers below from init_data_ and returns its solution
    // (the last iterate when it does not converge).
    Eigen::VectorXd solve(
        casadi::Function& solver,
        casadi::DM& lam_x,
        const casadi::DM& params
    );
    void filter_adjacent_collision_pairs();

    void generalize_ext_wrenches(
        const Eigen::VectorXd* EE_efrc_L = nullptr,
        const Eigen::VectorXd* EE_efrc_R = nullptr
    );

    std::pair<double, double> compute_error(Eigen::Matrix4d, Eigen::Matrix4d);
    pinocchio::Model model_;
    pinocchio::Data data_;
    pinocchio::GeometryModel geom_model_;
    pinocchio::GeometryData geom_data_;

    // nlpsol functions built once: x = q, p = [q_last; vec(target(s))].
    casadi::Function both_solver_;
    casadi::Function left_solver_;
    casadi::Function right_solver_;
    casadi::DM lower_limits_;
    casadi::DM upper_limits_;
    // Bound multipliers of each solver's last solution, for its warm start.
    casadi::DM both_lam_x_;
    casadi::DM left_lam_x_;
    casadi::DM right_lam_x_;

    pinocchio::FrameIndex L_hand_id_;
    pinocchio::FrameIndex R_hand_id_;

    int nq_;
    int nv_;

    pinocchio::FrameIndex oMcamera;
    pinocchio::FrameIndex oMLidar;

    Eigen::VectorXd init_data_;

    Eigen::VectorXd sol_q;
    Eigen::VectorXd sol_v;
    Eigen::VectorXd sol_t;
    Eigen::VectorXd sol_t_ext_L;
    Eigen::VectorXd sol_t_ext_R;

    pinocchio::Data::Matrix6x J_L;
    pinocchio::Data::Matrix6x J_R;

};

} // HumanoidPilot namespace
#endif // ROBOT_ARM_IK_H