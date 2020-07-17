#pragma once

#include <ceres/ceres.h>

class OdometryFactor : public ceres::SizedCostFunction<6, 7, 7, 7, 2>
{
    public:

    OdometryFactor() = delete;

    OdometryFactor( double _odo_accum_l, double _odo_accum_r )
    {
        odo_accum_l_ = _odo_accum_l;
        odo_accum_r_ = _odo_accum_r;
    }

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Eigen::Vector3d Pi( parameters[0][0], parameters[0][1], parameters[0][2] );
        Eigen::Quaterniond Qi( parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5] );

        Eigen::Vector3d Pj( parameters[1][0], parameters[1][1], parameters[1][2] );
        Eigen::Quaterniond Qj( parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5] );

        Eigen::Vector3d T_ob( parameters[2][0], parameters[2][1], parameters[2][2] );
        Eigen::Quaterniond C_ob( parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5] );

        double r_l = parameters[3][0];
        double r_r = parameters[3][0];
        double d_inv = parameters[3][1];

        Eigen::Map<Eigen::Matrix<double, 6, 1>> residual( residuals );

        double odo_rot_angle = ( odo_accum_l_ * r_l - odo_accum_r_ * r_r ) * d_inv;
        Eigen::AngleAxisd odo_rot_vec( odo_rot_angle, Eigen::Vector3d( 0., 0., 1. ) );
        Eigen::Matrix3d odo_rot_mat = odo_rot_vec.toRotationMatrix();   // cos-sin matrix
        Eigen::Matrix3d res_rot_mat = C_ob.toRotationMatrix().inverse() * odo_rot_mat * C_ob * \
                                        Qj.toRotationMatrix().inverse() * \
                                        Qi.toRotationMatrix();
        Eigen::AngleAxisd res_rot_vec( res_rot_mat );
        // TODO
        residual.template block<3,1>( 3, 0 ) = res_rot_vec.angle() * res_rot_vec.axis();    // 1. Pos; 2. Rot
        double odo_trans_len = ( odo_accum_l_ * r_l + odo_accum_r_ * r_r ) * 0.5;
        Eigen::Vector3d odo_trans_vec;
        if( fabs(odo_rot_angle) <= 1e-6 ) odo_trans_vec << odo_trans_len, 0., 0.;
        else odo_trans_vec << odo_trans_len / odo_rot_angle * sin(odo_rot_angle), odo_trans_len / odo_rot_angle * (1-cos(odo_rot_angle)), 0.;
        Eigen::Vector3d imu_trans_o_frame = ( odo_rot_mat - Eigen::Matrix3d::Identity() ) * T_ob + odo_trans_vec;                                
        residual.template block<3,1>(0,0) = \
            C_ob.toRotationMatrix().inverse() * imu_trans_o_frame - Qi.toRotationMatrix().inverse() * ( Pj - Pi );

        residual = residual * 10;
/*
        std::cout << "Odo accums : " << odo_accum_l_ << " , " << odo_accum_r_ << std::endl;
        std::cout << "Body Pos: " << std::endl;
        std::cout << Pi.transpose() << std::endl << Pj.transpose() << std::endl;
        std::cout << "Residual : " << std::endl;
        std::cout << residual.transpose() << std::endl;

       std::cout << "odo trans length : \n" << odo_trans_vec.transpose() << std::endl;
       */

        if( jacobians )
        {
            if( jacobians[0] )
            {
                Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> jacobian_pose_i( jacobians[0] );
                jacobian_pose_i.setZero();

                jacobian_pose_i.template block<3,3>( 0, 0 ) = -1 * Qi.toRotationMatrix();
                jacobian_pose_i.template block<3,3>( 3, 3 ) = -1 * Qi.toRotationMatrix().inverse() * Qj.toRotationMatrix();   // Ignore J

                jacobian_pose_i = jacobian_pose_i * 10;
            }
            if( jacobians[1] )
            {
                Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> jacobian_pose_j( jacobians[1] );
                jacobian_pose_j.setZero();

                Eigen::Vector3d imu_trans_b_frame = Qi.toRotationMatrix().inverse() * ( Pj - Pi );
                jacobian_pose_j.template block<3,3>( 0, 3 ) << 0., -imu_trans_b_frame(2), imu_trans_b_frame(1), \
                                                                                                                imu_trans_b_frame(2), 0., -imu_trans_b_frame(0), \
                                                                                                                -imu_trans_b_frame(1), imu_trans_b_frame(0), 0.;
                jacobian_pose_j.template block<3,3>( 0, 3 ) = jacobian_pose_j.template block<3,3>( 0, 3 ) * -1;
                jacobian_pose_j.template block<3,3>( 0, 0 ) = Qi.toRotationMatrix().inverse();
                jacobian_pose_j.template block<3,3>( 3, 3 ) = Eigen::Matrix3d::Identity();  // Ignore J

                jacobian_pose_j = jacobian_pose_j * 10;
            }
            if( jacobians[2] )
            {
                Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> jacobian_T_ob( jacobians[2] );
                jacobian_T_ob.setZero();

                Eigen::Vector3d odo_rot_b_frame = C_ob.toRotationMatrix().inverse().block<3,1>(0, 2) * odo_rot_angle;
                Eigen::Matrix3d skew_odo_rot_mat;
                skew_odo_rot_mat << 0., -odo_rot_b_frame(2), odo_rot_b_frame(1), \
                                                                odo_rot_b_frame(2), 0., -odo_rot_b_frame(0), \
                                                                -odo_rot_b_frame(1), odo_rot_b_frame(0), 0.;
                jacobian_T_ob.template block<3,3>( 3, 3 ) = Qi.toRotationMatrix().inverse() * Qj.toRotationMatrix() * skew_odo_rot_mat;
                Eigen::Vector3d imu_trans_b_frame_2 = C_ob.toRotationMatrix().inverse() * imu_trans_o_frame;
                jacobian_T_ob.template block<3,3>( 0, 3 ) << 0., -imu_trans_b_frame_2(2), imu_trans_b_frame_2(1), \
                                                                                imu_trans_b_frame_2(2), 0., -imu_trans_b_frame_2(0), \
                                                                                -imu_trans_b_frame_2(1), imu_trans_b_frame_2(0), 0.;
                jacobian_T_ob.template block<3,3>( 0, 0 ) = C_ob.toRotationMatrix().inverse() * ( odo_rot_mat - Eigen::Matrix3d::Identity() );

                jacobian_T_ob = jacobian_T_ob * 10;
            }
            if( jacobians[3] )
            {
                Eigen::Map<Eigen::Matrix<double, 6, 2, Eigen::RowMajor>> jacobian_r_b( jacobians[3] );
                jacobian_r_b.setZero();

                Eigen::Vector3d J;
                if( fabs(odo_rot_angle) < 1e-6 ) J << 1., 0., 0.;
                else J << sin(odo_rot_angle)/odo_rot_angle, (1-cos(odo_rot_angle))/odo_rot_angle, 0.;

                Eigen::Matrix3d skew_z;
                skew_z << 0., -1., 0., 1., 0., 0., 0., 0., 0.;
                jacobian_r_b.template block<3,1>( 0, 0 ) = C_ob.toRotationMatrix().inverse() * ( J*(odo_accum_l_+odo_accum_r_)/2. + \
                                                                                                                                                                                            (odo_accum_l_-odo_accum_r_)*d_inv*skew_z*T_ob );
                jacobian_r_b.template block<3,1>( 3, 0 ) = Qi.toRotationMatrix().inverse() * Qj.toRotationMatrix() * \
                                                                                                        C_ob.toRotationMatrix().inverse().block<3,1>(0,2) * (odo_accum_l_ - odo_accum_r_) * d_inv;
                jacobian_r_b.template block<3,1>( 0, 1 ) = C_ob.toRotationMatrix().inverse() * ( (odo_accum_l_-odo_accum_r_)*r_l*skew_z*T_ob );
                jacobian_r_b.template block<3,1>( 3, 1 ) = Qi.toRotationMatrix().inverse() * Qj.toRotationMatrix() * \
                                                                                                        C_ob.toRotationMatrix().inverse().block<3,1>(0,2) * (odo_accum_l_ - odo_accum_r_) * r_l;

                jacobian_r_b = jacobian_r_b * 10;
            }
        }

        return true;
    }

    double odo_accum_l_, odo_accum_r_;
    
    
};