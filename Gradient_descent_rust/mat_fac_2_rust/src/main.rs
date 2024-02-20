use ndarray::{Array2, Axis};
use rand::Rng;

use std::error::Error;
use std::fs::File;
use std::path::Path;

use csv::ReaderBuilder;
use ndarray_rand::rand_distr::Uniform;

use std::io::Write;


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

fn load_movielens(path: &str) -> Array2<f64> {
    let file = File::open(path).expect("Error opening file");
    let mut rdr = ReaderBuilder::new().delimiter(b'\t').from_reader(file);
    
    let num_users = 943;
    let num_items = 1682;

    // Create a matrix initialized with zeros
    let mut matrix = Array2::zeros((num_users, num_items));

    // Fill the matrix with ratings from the u.data file
    for result in rdr.records() {
        let record = result.expect("Error reading record");
        let user_id: usize = record[0].parse().expect("Error parsing user_id");
        let item_id: usize = record[1].parse().expect("Error parsing item_id");
        let rating: f64 = record[2].parse().expect("Error parsing rating");

        // Check if the user and item IDs are within the expected range
        if user_id <= num_users && item_id <= num_items {
            // Set the rating in the matrix (subtracting 1 from IDs to match array indexing)
            matrix[[user_id - 1, item_id - 1]] = rating;
        }
    }

    matrix
}

fn gradient_descent(
    r: &Array2<f64>,
    p: &mut Array2<f64>,
    v: &mut Array2<f64>,
    latent_features: usize,
    regularization_parameter: f64,
    learning_rate: f64,
    max_epochs: usize,
    convergence_threshold: f64,
) -> (Array2<f64>, Array2<f64>, Vec<f64>) {
    let m = r.nrows();
    let n = r.ncols();
    let k = latent_features;

    let mut error_list = Vec::new(); //new vector for errors

    let mut i_list: Vec<(usize, usize)> = Vec::new();

    // Iterate over the rows and columns of R
    for i in 0..m {
        for j in 0..n {
            // Check if R[i, j] is non-zero
            if r[[i, j]] != 0.0 {
                // Add tuple (i, j) to i_list
                i_list.push((i, j));
            }
        }
    }


    let mut r_hat: Array2<f64> = Array2::zeros((m, n));

    let mut epoch = 1;
    let mut mae = 10.0;

    while epoch < max_epochs && mae >= convergence_threshold {

    // Calculate the error matrix
        let error_matrix = r - &r_hat;

        mae = mean_absolute_error(&r, &r_hat);

        error_list.push(mae); // Save error for the current epoch
        
        println!("MAE at Epoch {}: {}", epoch, mae);


        //Update matrices P, V
        for &(i, j) in i_list.iter() {
            for l in 0..k {
                let sum_term_1 = error_matrix[[i, j]] * v[[l, j]];
                let sum_term_2 = error_matrix[[i, j]] * p[[i, l]];
        
                p[[i, l]] += learning_rate * sum_term_1 + regularization_parameter * p[[i, l]];
                v[[l, j]] += learning_rate * sum_term_2 + regularization_parameter * v[[l, j]];
            }
        }

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
    // // Provide the path to your MovieLens 100k dataset file
    // let movielens_path = "/Users/Jolijn/projects/svd-using-gradient-descent/ml-100k/u.data";

    // // Load the dataset
    // let r = load_movielens(movielens_path);

    // let m = r.nrows();
    // let n = r.ncols();

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

    for _ in 0..num_simulations {
        let r = create_sparse_matrix(m, n, density);

        println!("Matrix R\n{:?}", r);

        let mut p = random_init((m, latent_features), scale);
        let mut v = random_init((latent_features, n), scale);

        println!("P:\n{:?}", p);
        println!("V:\n{:?}", v);

        let (p_final, v_final, error_list) = gradient_descent(
            &r,
            &mut p,
            &mut v,
            latent_features,
            regularization_parameter,
            learning_rate,
            max_epochs,
            convergence_threshold,
        );

        // //Write the error_list to a file
        // if let Ok(mut file) = File::create("error_list.txt") {
        //     for error in &error_list {
        //         writeln!(file, "{}", error).expect("Error writing to file");
        //     }
        //     println!("Error list written to error_list.txt");
        // } else {
        //     println!("Error creating or writing to error_list.txt");
        // }

        // Check if any error in the error list is less than the convergence threshold
        if error_list.iter().any(|&error| error < convergence_threshold) {
            converged_count += 1;
        }

        // println!("Final P:\n{:?}", p_final);
        // println!("Final V:\n{:?}", v_final);

        let r_hat_final = p_final.dot(&v_final);
        // println!("Final R_hat (P * V):\n{:?}", r_hat_final);

        let error_matrix = &r - &r_hat_final;

        let mae = mean_absolute_error(&r, &r_hat_final);

        println!("Mean absolute error {}", mae);
    }

    println!("Converged in {}/{} simulations", converged_count, num_simulations);
}