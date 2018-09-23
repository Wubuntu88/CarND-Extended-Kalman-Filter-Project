#include <iostream>
#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
    * predicts the state
  */

  x_ = F_ * x_;

  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {

  VectorXd y(2);
  MatrixXd S(2, 2);
  MatrixXd K(4, 2);

  y = z - (H_ * x_);
  S = H_ * P_ * H_.transpose() + R_;
  K = P_ * H_.transpose() * S.inverse();

  x_ = x_ + (K * y);

  MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());

  P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  float px = x_(0);
  float py = x_(1);
  float vx = x_(2);
  float vy = x_(3);

  // Lesson 5, part 14
  float rho = sqrt(px*px + py*py);
  float theta = atan2(py,px);
  float rho_dot = (px*vx + py*vy) / rho;

  VectorXd z_pred = VectorXd(3);
  z_pred << rho, theta, rho_dot;

  // Lesson 5, part 7
  VectorXd y_vector = z - z_pred;

  MatrixXd H_P_Ht = MatrixXd(4, 4);
  MatrixXd S = MatrixXd(3, 3);
  MatrixXd K = MatrixXd(4, 3);
  MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());

  H_P_Ht = H_ * P_ * H_.transpose();
  S = H_P_Ht + R_;
  K = P_ * H_.transpose() * S.inverse();

  x_ = x_ + (K * y_vector);

  P_ = (I - K * H_) * P_;
}
