// Algorithm 3a: Matrix factorization using gradient descent

use std::fs::File;
use std::io::Write;

use ndarray::{Array2}; // Importing necessary modules from ndarray
use rand::Rng; // Importing Rng trait from rand

fn random_init(shape: (usize, usize), scale: f64) -> Array2<f64> {
    let mut rng = rand::thread_rng();
    Array2::from_shape_fn(shape, |_| rng.gen_range(0.0..scale))
}

fn create_sparse_matrix(m: usize, n: usize, density: f64) -> Array2<f64> {
    let mut sparse_matrix = Array2::from_elem((m, n), 0.0);

    for i in 0..m {
        for j in 0..n {
            if rand::random::<f64>() < density {
                sparse_matrix[[i, j]] = rand::random::<f64>() * 5.0 + 1.0;
                sparse_matrix[[i, j]] = sparse_matrix[[i, j]].floor();
            }
        }
    }

    sparse_matrix
}

fn mean_absolute_error(matrix1: &Array2<f64>, matrix2: &Array2<f64>) -> f64 {
    let error = matrix1 - matrix2;
    let abs_error = error.mapv(|x| x.abs());
    abs_error.mean().unwrap()
}

fn gradient_descent(
    r: &Array2<f64>,
    p: &mut Array2<f64>,
    v: &mut Array2<f64>,
    regularization_parameter: f64,
    learning_rate: f64,
    max_epochs: usize,
    convergence_threshold: f64,
) -> (Array2<f64>, Array2<f64>, Vec<f64>) {
    let m = r.nrows();
    let n = r.ncols();

    let mut error_list = Vec::new(); //new vector to store errors during each epoch

    let mut r_hat = p.dot(v);

    let mut epoch = 0;
    let mut mae = 10.0; //Random value for initialization

    while epoch < max_epochs && mae >= convergence_threshold {
        // Calculate the error matrix
        //let error_mat = r - &r_hat;

        mae = mean_absolute_error(&r, &r_hat);

        error_list.push(mae); // Save error for the current epoch
        
        println!("MAE at Epoch {}: {}", epoch, mae);
        
        // Derivatives with respect to matrices P and V
        let term1 = r_hat.clone() - r;
        let term2 = regularization_parameter * p.clone();
        let dldp = term1.clone().dot(&v.t()) + term2;

        let term3 = r_hat.clone() - r;
        let term4 = term3.t();
        let term5 = regularization_parameter * v.clone();
        let term6 = term5.t();
        let dldv = term4.clone().dot(p) + term6;

        // Updating matrices P, V
        p.scaled_add(-learning_rate, &dldp);
        v.scaled_add(-learning_rate, &dldv.t());

        r_hat = p.dot(v);

        epoch += 1;
    }

    if mae < convergence_threshold {
        // Converged
        println!("Converged at epoch: {}", epoch-1);
    }

    (p.clone(), v.clone(), error_list)
}

fn main() {
    let num_simulations = 100;
    let convergence_threshold = 0.2;

    let m = 3; // Number of rows
    let n = 3; // Number of columns
    let density = 1.0; // Desired density (adjust as needed)

    let learning_rate = 0.01;
    let regularization_parameter = 0.0;
    let max_epochs = 1000;
    let latent_features = 2;

    let scale = 1.0; // Scale for random initialization

    let mut converged_count = 0;

    for _ in 0..num_simulations{
    
        let r = create_sparse_matrix(m, n, density);
        //println!("Matrix R:\n{:?}", r);

        let mut p = random_init((m, latent_features), scale);
        let mut v = random_init((latent_features, n), scale);

        //println!("Initial matrix P:\n{:?}", p);
        //println!("Initial matrix V:\n{:?}", v);

        let (p_final, v_final, error_list) = gradient_descent(
            &r,
            &mut p,
            &mut v,
            regularization_parameter,
            learning_rate,
            max_epochs,
            convergence_threshold,
        );

        // // Write the error_list to a file
        // if let Ok(mut file) = File::create("error_list_.txt") {
        //     for error in &error_list {
        //         writeln!(file, "{}", error).expect("Error writing to file");
        //     }
        //     println!("Error list written to error_list_.txt");
        // } else {
        //     println!("Error creating or writing to error_lis_.txt");
        // }

        // Check if any error in the error list is less than the convergence threshold
        if error_list.iter().any(|&error| error < convergence_threshold) {
            converged_count += 1;
        }

        println!("Final P:\n{:?}", p_final);
        println!("Final V:\n{:?}", v_final);

        let r_hat_final = p_final.dot(&v_final);
        let error_matrix = &r - &r_hat_final;

        let mae = mean_absolute_error(&r, &r_hat_final);
        println!("Mean absolute error {}", mae);

    }
    println!("Converged in {}/{} simulations", converged_count, num_simulations);

}
