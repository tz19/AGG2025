function R = zxz_rotation_matrix(eul)
    % Convert angles to radians
    alpha = deg2rad(eul(1));
    beta = deg2rad(eul(2));
    gamma = deg2rad(eul(3));

    % Calculate rotation matrices around z, x, and z axes
    Rz_alpha = [cos(alpha) -sin(alpha) 0; sin(alpha) cos(alpha) 0; 0 0 1];
    Rx_beta = [1 0 0; 0 cos(beta) -sin(beta); 0 sin(beta) cos(beta)];
    Rz_gamma = [cos(gamma) -sin(gamma) 0; sin(gamma) cos(gamma) 0; 0 0 1];

    % Calculate the overall rotation matrix
    R = Rz_gamma * Rx_beta * Rz_alpha;
end