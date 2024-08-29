extern crate nalgebra as na;
use na::{DMatrix, DVector, Matrix3, Vector3};

type Vec3 = Vector3<f64>;
type Mat3 = Matrix3<f64>;
let g = 9.81;

fn skew(x: Vec3) -> Mat3 {
    Mat3::new(0.0, -x[2], x[1], x[2], 0.0, -x[0], -x[1], x[0], 0.0)
}

fn check_inside_pyramid(ay_max: Vec3, yaw_max: Vec3, long_max: Vec3, state_vec: Vec3) -> f64 {
    let dir_sign = long_max[0].signum();
    let k = long_max - state_vec;
    let a = na::dot(k * na::dot(skew(long_max - yaw_max),  (ay_max - yaw_max))) * dir_sign * -1.0;
    let b = na::dot(k * na::dot(skew(long_max - yaw_max), -(ay_max - yaw_max))) * dir_sign;
    let c = na::dot(k * na::dot(skew(long_max + yaw_max),  (ay_max + yaw_max))) * dir_sign;
    let d = na::dot(k * na::dot(skew(long_max + yaw_max), -(ay_max + yaw_max))) * dir_sign * -1.0;
    let e = state_vec[0] * dir_sign * -1.0;
    let m = max([a, b, c, d, e]);
    m
}

fn calc_vel(c_0: f64, c_1: f64, c_2: f64, v_min: f64 = 0.1) -> (f64, bool) {
    // Solve the quadratic equation for the velocity
    let v_1 = (c_1.powi(2) - (4.0 * c_2 * c_0)).sqrt() - c_1 / (2.0 * c_2);
    let v_2 = (((c_1.powi(2) - (4.0 * c_2 * c_0)).sqrt()) * -1.0) - c_1 / (2.0 * c_2);
    let v_0 = v_1.max(v_2);
    if v_0 > v_min {
        (v_0, false)
    }
    (v_min, true)
}

fn check_ind(j: i32, k: i32, ind: i32, u_lend: i32, asc: bool) -> (i32, i32, i32) {
    let (j, k, ind) = if asc {
        if ind >= u_lend || ind <= -u_lend {
            (0, 0, 0)
        } else if k + ind >= u_lend {
            (j, -ind, ind)
        } else if j + ind >= u_lend {
            (-ind, k, ind)
        } else if j == k {
            (j, k + 1, ind)
        } else {
            (j, k, ind)
        }
    } else {
        if ind >= u_lend || ind <= -u_lend {
            (0, 0, 0)
        } else if ind - k <= -u_lend {
            (j, ind - u_lend + 1, ind)
        } else if ind - j <= -u_lend {
            (ind - u_lend + 1, k, ind)
        } else if j == k {
            (j, k - 1, ind)
        } else {
            (j, k, ind)
        }
    };
    (j, k, ind)
}

fn solve_point(aymax: Vec3, yawmax: Vec3, longacc: Vec3, v_k: f64, v_j: f64, ds: f64, k_k: f64, k_j: f64, vbp: f64, vbb: f64) -> (f64, f64, f64, f64, f64) {
    let point = longacc; // arbitrary point on LAS
    // Make the LAS data structures
    let mut v_it: [f64; 4] = [0.0; 4];
    let mut longacc_it: [f64; 4] = [0.0; 4];
    let mut latacc_it: [f64; 4] = [0.0; 4];
    let mut omegadot_it: [f64; 4] = [0.0; 4];
    let mut d_check: [f64; 4] = [0.0; 4];
    let mut latacc: [Vec3; 4] = [Vec3::new(); 4];
    let mut yawacc: [Vec3; 4] = [Vec3::new(); 4];
    latacc[0] = aymax; // latacc on Bottom-Right Corner
    latacc[1] = aymax; // latacc on Bottom-Right Corner
    latacc[2] = -aymax; // latacc on Top-Left Corner
    latacc[3] = -aymax; // latacc on Top-Left Corner
    yawacc[0] = yawmax; // yawacc on Top-Right Corner
    yawacc[1] = -yawmax; // yawacc on Bottom-Left Corner
    yawacc[2] = yawmax; // yawacc on Top-Right Corner
    yawacc[3] = -yawmax; // yawacc on Bottom-Left Corner
    // Calculate the velocity limit based on the longitudinal acceleration limits
    let max_vel = longacc * ds + v_k;
    // If we are limited by max power output use that as the velocity limit
    if vbp < max_vel {
        max_vel = vbp;
    }
    if vbb < max_vel {
        max_vel = vbb;
    }
    // Check for different LAS for solution to roots
    let nvec = array![1.0/g, 1.0/g, 1.0/10.0];
    for i in 0..4 {
        let n_vector = np.dot(skew(longacc - yawacc[i, :]), latacc[i, :] - yawacc[i, :]) // Normal vector to LAS
        let c_0 = n_vector[0] * (point[0] + (v_k.ipow(2) / (2 * ds))) + (n_vector[1] * (point[1] - ((v_k.ipow(2) * (k_j + k_k)) / 8))) + n_vector[2] * (point[2] + (v_k.ipow(2) * k_k / (2 * ds)))
        let c_1 = v_k * (-n_vector[1] * ((k_k + k_j) / 4) + n_vector[2] * ((k_k - k_j) / (2 * ds)))
        let c_2 = -n_vector[0] / (2 * ds) - n_vector[1] * (k_k + k_j) / 8 - n_vector[2] * k_j / (2 * ds)
        
        let v_save, nps = calc_vel(c_0, c_1, c_2)
        if nps or v_save > max_vel:
            v_save = max_vel
        if v_save < v_j:
            v_it[i] = v_save
        else:
            v_it[i] = v_j
            
        longacc_it[i] = (v_it[i].ipow(2) - v_k.ipow(2)) / (2 * ds) // equation 116
        latacc_it[i] = ((k_k + k_j) / 2) * (((v_k + v_it[i]) / 2).ipow(2)) // equation 117
        omegadot_it[i] = ((v_k + v_it[i]) / (2 * ds)) * ((k_j * v_it[i]) - (k_k * v_k)) // equation 118
        let a_check = array!([longacc_it[i], latacc_it[i], omegadot_it[i]])
        // Check if the solution is inside the LAS
        let d_check[i] = check_inside_pyramid(aymax*nvec, yawmax*nvec, longacc*nvec, a_check*nvec)
    }
    let good_ind = d_check.iter().enumerate().filter(|&(_, &v)| v < 0.1).map(|(i, _)| i).collect::<Vec<_>>();
    if !good_ind.any() {
        let d_check_min = d_check.iter().enumerate().min_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap()).unwrap();
        if longacc_it[d_check_min].abs() < 0.05 {
            (v_it[d_check_min], latacc_it[d_check_min], longacc_it[d_check_min], omegadot_it[d_check_min], d_check[d_check_min])
        } else {
            let new_v = (v_it[best_ind] + v_k) / 2;
            (new_v, 0.0, 0.0, 0.0, d_check[d_check_min])
        }
    } else {
        // Find the good ind with the highest velocity
        let best_ind = good_ind.iter().enumerate().max_by(|&(_, a), &(_, b)| v_it[*a].partial_cmp(&v_it[*b]).unwrap()).unwrap();
        (v_it[best_ind], latacc_it[best_ind], longacc_it[best_ind], omegadot_it[best_ind], d_check[best_ind])
    }
}

fn interp_las_corner(vel: f64, vels: &Vec<f64>, point_arr: &Vec<Vec3>) -> Vec3 {
    // find the velocity bin above and below the current velocity
    if vel > vels[vels.len() - 1] {
        point_arr[vels.len() - 1]
    } else if vel < vels[0] {
        point_arr[0]
    }
    let mut upper_ind = 0;
    for (i, v) in vels.iter().enumerate() {
        if *v > vel {
            upper_ind = i;
            break;
        }
    }
    let lower_ind = (upper_ind - 1).max(0);
    // Interpolate the point
    let upper_weight = (vel - vels[lower_ind]) / (vels[upper_ind] - vels[lower_ind]);
    let lower_weight = 1.0 - upper_weight;
    let x = point_arr[upper_ind][0] * upper_weight + point_arr[lower_ind][0] * lower_weight;
    let y = point_arr[upper_ind][1] * upper_weight + point_arr[lower_ind][1] * lower_weight;
    let z = point_arr[upper_ind][2] * upper_weight + point_arr[lower_ind][2] * lower_weight;
    Vec3::new(x, y, z)
}

fn find_crits(vels: Vec<f64>) -> Vec<usize> {
    let crit_diff = vels[..vels.len() - 1].iter().zip(vels[1..].iter()).map(|(a, b)| a - b).map(|a| a.signum()).collect::<Vec<_>>();
    let critd_it = crit_diff[..crit_diff.len() - 1].iter().zip(crit_diff[1..].iter()).map(|(a, b)| a + b).enumerate().filter(|(_, a)| *a == 0).map(|(i, _)| i).collect::<Vec<_>>();
    let critc = (vec![0]).into_iter().chain(critd_it.into_iter()).collect::<Vec<_>>();
    critc.into_iter().map(|a| vels[a]).collect::<Vec<_>>()
}

fn find_vel_limit(vel: f64, vels: Vec<f64>, aymax: Vec<Vec3>, yawmax: Vec<Vec3>, k: f64, k_prime: f64, u: f64, itt: i64) -> (f64, f64) {
    if vel.is_nan() {
        (vel, 0.0)
    }
    let a_0 = interp_LAS_corner(vel, vels, aymax);
    let a_1 = interp_LAS_corner(vel, vels, yawmax);
    if k == 0.0 {
        (std::f64::INFINITY, 0.0) // if there is no curvature then return inf velocity limit
    }
    let mut lsbc_check = false;
    let mut chk_count = 0;
    while (lsbc_check == false) {
        if chk_count == 1 {
            a_0[1] *= -1.0;
        } else if chk_count == 2 {
            a_1[1] *= -1.0;
        } else if chk_count == 3 {
            a_0[1] *= -1.0;
        } else if chk_count == 4 {
            lsbc_check = true;
        }
        let num = a_0[1] - ((a_1[1] * (a_0[0] * k - a_0[2])) / ((a_1[0] * k) - a_1[2]));
        let denom = k + (a_1[1] * (k_prime) / ((a_1[0] * k) - a_1[2]));
        if (num / denom) < 0 {
            chk_count += 1;
            continue;
        }
        let vel_limit = (num / denom).sqrt();
        lsbc_check = true;
    }
    if vel_limit.is_nan() {
        (std::f64::INFINITY, ((a_1[2] * u) / (k * 2)).abs().sqrt())
    }
    if (vel_limit - vel).abs() < 0.1 || itt > 10 {
        (vel_limit, a_1[2])
    } else {
        find_vel_limit(vel_limit, vels, aymax, yawmax, k, k_prime, u, itt+1)
    }
}

fn sim_ian(aymax: Vec<Vec3>, yawmax: Vec<Vec3>, longacc_forward: Vec<Vec3>, longacc_reverse: Vec<Vec3>, vels: Vec<f64>, k: Vec<f64>, k_prime: Vec<f64>, u: Vec<f64>, v_init: f64, target, plot = False, flying=False, silent=False) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    // Line Segment of LSBC (Bottom-Right of MMD) (Patton, 74)
    let middle_vel = (vels[0] + vels[vels.len() - 1]) / 2.0;

    // Solving for initial velocity profile [m/s] (Patton, 75)
    // Checks for each segment along LSBC
    // This does not include beta double prime
    let u_len = u.len();
    let vel_begin = time::Instant::now();
    let mut velocity = Vec<f64>::with_capacity(u_len);
    let mut velocity_curve = Vec<f64>::with_capacity(u_len);
    let mut velocity_curve_rate = Vec<f64>::with_capacity(u_len);
    for i in 1..u_len {
        (velocity_curve[i], velocity_curve_rate[i]) = find_vel_limit(middle_vel, vels, aymax, yawmax, (k[i] + k[i-1]) / 2.0,  (k_prime[i] + k_prime[i - 1]) / 2.0, (u[i] - u[i - 1]));
        velocity[i] = f64::min(velocity_curve[i], velocity_curve_rate[i]);
    }
    
    
    if !silent {
        println!("Velocity Profile Calculation: {}s", time::Instant::now().duration_since(vel_begin).as_secs_f64());
    }
    
    velocity[velocity <= 0] = velocity[velocity > 0].min()
    if !flying {
        velocity[0] = v_init;
    }
        
    let mut long_g = Vec<f64>::with_capacity(u_len);
    if flying {
        long_g[0]  = (velocity[1] - velocity[-1]) / ((2 * u[1]) / ((velocity[1] + velocity[-1]) / 2.0)) / g; // Connecting the long accel  of the track back to the beginning (flying lap)
    }

    long_g[-1] = (velocity[-2] - velocity[-1]) / ((u[-2] - u[-1]) / ((velocity[-2] + velocity[-1]) / 2.0)) / g; // Setting the long accel at the  so that it bls to the start of the track

    // Creating long accel map using delta V/delta X

    long_g[1..-1] = (velocity[2..] - velocity[..-2]) / ((track.u[2..] - track.u[..-2]) / ((velocity[2..] + velocity[..-2]) / 2)) / g;

    let mut critc = find_crits(velocity);

    let dd = u[2] - u[0];
    
    // Initialize counters and other variables for iteration
    let mut k, j, n, v_error = 0, 1, 1, 1;


    // Initialize vectors for accelerations in iteration
    // Initialize D vector in iteration (Patton, 78)
    let mut v_it = velocity.clone();
    latacc_it, longacc_it, omegadot_it, d_fcheck, d_rcheck = np.zeros([u_len]), np.zeros([u_len]), np.zeros([u_len]), np.zeros([u_len]), np.zeros([u_len])

    err, cc, lt = [], [], []
    let vel_begin = time::Instant::now();
    let mut last = begin;
    if not silent:
        print(f"Num_points: {v_it.shape[0]}")

    let mut dt = Vec<f64>::with_capacity(u_len);
    while (v_error > target) {
        v_it_old = v_it.copy()
        checks = 0
        // Limit velocities are calculated by starting from critical points and
        // identifying the next velocity based on acceleration limits (LAS of Octahedron)
        for i, ind in enumerate(critc):
            // Working from the front side

            j, k, ind = check_ind(1, 0, ind, u_len, true)
            while v_it[ind + k] <= v_it[ind + j]:
                if (((ind + j + 2 * u_len) % u_len) - ((ind + k + 2 * u_len) % u_len)) != 1 and not flying:
                    break
                checks += 1
                ds = u[ind + j] - u[ind + k]
                if abs(ds) > abs(dd):
                    ds = dd
                vv = (v_it[ind + k] + v_it[ind + j]) / 2
                vb_power = np.sqrt(vv**2 + 2 * (car.find_tractive_force(vv) / car.mass) * ds)
                v_it[ind+j], latacc_it[ind+j], longacc_it[ind+j], omegadot_it[ind+j], d_fcheck[ind+j] = solve_point(interp_LAS_corner(vv, vels, aymax), interp_LAS_corner(vv, vels, yawmax), interp_LAS_corner(vv, vels, longacc_forward), v_it[ind+k], v_it[ind+j], ds, k[ind + k], k[ind + j], vbp=vb_power)

                k += 1
                j += 1
                j, k, ind = check_ind(j, k, ind, u_len, true)
            
            j, k, ind = check_ind(1, 0, ind, u_len, false)
            // Working from the back side
            while v_it[ind-k] <= v_it[ind-j]:
                if (((ind - j + 2 * u_len) % u_len) - ((ind - k + 2 * u_len) % u_len)) != -1 and not flying:
                    break
                checks += 1

                ds = -(u[ind - k] - u[ind - j])
                if abs(ds) > abs(dd):
                    ds = -dd
                    
                vv = (v_it[ind - k] + v_it[ind - j]) / 2
                // vb_break = np.sqrt(v_it[ind - k]**2 + 2 * (car.find_tractive_force_braking(v_it[ind - k]) / car.mass) * ds)
                v_it[ind-j], latacc_it[ind-j], longacc_it[ind-j], omegadot_it[ind-j], d_rcheck[ind-j] = solve_point(interp_LAS_corner(vv, vels, aymax), interp_LAS_corner(vv, vels, yawmax), interp_LAS_corner(vv, vels, longacc_reverse), v_it[ind-k], v_it[ind-j], ds, k[ind - k], k[ind - j]) # , vbb=vb_break

                k += 1
                j += 1
                j, k, ind = check_ind(j, k, ind, u_len, false)
                
        v_error_list = (v_it_old - v_it).abs() / v_it;
        v_error = np.nanmax(v_error_list)
        
        critc = find_crits(v_it);
        if !silent {
            dt[1..] = 2.0 * (u[1..] - u[..-1]) / (v_it[1..] + v_it[..-1]);
            totallen = np.sum(dt)
            println!("{}\ttime_el:{:.2}\ttime_last:{:.4}\tv_err: {:.6}\tcc: {}\titt:{}\ttime:{:.2}", n, (time::Instant::now() - begin).as_secs_f64(), (time::Instant::now() - last).as_secs_f64(), v_error, critc.len(), checks, totallen);
        }

        last = time::Instant::now();
        checks = 0
        n += 1;
    }

    if !silent {
        println!("total time: {:.2}", (time::Instant::now() - begin).as_secs_f64());
    }

    // v_it = np.where(v_it < 0.1, 0.1, v_it)

    
    dt[1..] = 2.0 * (u[1..] - u[..-1]) / (v_it[1..] + v_it[..-1]);

    (longacc_it, latacc_it, omegadot_it, dt, long_g, v_it, velocity, critc, d_fcheck, d_rcheck, velocity_curve, velocity_curve_rate)
}