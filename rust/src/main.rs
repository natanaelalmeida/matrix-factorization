use sprs::{CsMat, CsVec, SparseMat};
use rayon::prelude::*;

struct MatrixFactorization {
    num_users: usize,
    num_items: usize,
    latent_factors: usize,
    learning_rate: f64,
    regularization: f64,
    user_factors: CsMat<f64>, // user x latent
    item_factors: CsMat<f64>, // item x latent
    global_bias: f64,
    user_bias: CsVec<f64>, // user
    item_bias: CsVec<f64>, // item
}

impl MatrixFactorization {
    fn new(num_users: usize, num_items: usize, latent_factors: usize, learning_rate: f64, regularization: f64) -> Self {
        let user_factors = CsMat::zero((num_users, latent_factors));
        let item_factors = CsMat::zero((num_items, latent_factors));
        let user_bias = CsVec::empty(num_users);
        let item_bias = CsVec::empty(num_items);
        let global_bias = 0.0;
        MatrixFactorization {
            num_users,
            num_items,
            latent_factors,
            learning_rate,
            regularization,
            user_factors,
            item_factors,
            global_bias,
            user_bias,
            item_bias,
        }
    }

    fn train(&mut self, mut ratings: &Vec<(usize, usize, f64)>, epochs: usize) {
        // Initialize user and item factors
        self.global_bias = ratings.iter().map(|x| x.2).sum::<f64>() / ratings.len() as f64;

        for epoch in 0..epochs {
            ratings.par_iter_mut().for_each(|(user, item, rating)| {
                let prediction = self.predict(*user, *item);
                let error = *rating - prediction;

                // Update biases
                *self.user_bias.get_mut(*user).unwrap() += self.learning_rate * (error - self.regularization * self.user_bias.get(*user).unwrap());
                *self.item_bias.get_mut(*item).unwrap() += self.learning_rate * (error - self.regularization * self.item_bias.get(*item).unwrap());

                // Update latent factors
                for k in 0..self.latent_factors {
                    let user_factor = self.user_factors.get(*user, k).unwrap();
                    let item_factor = self.item_factors.get(*user, k).unwrap();

                    self.user_factors[[user, *k]] += self.learning_rate * (error * item_factor - self.regularization * user_factor);
                    self.item_factors[[item, *k]] += self.learning_rate * (error * user_factor - self.regularization * item_factor);
                }
            })
        }

    }

    fn predict(&self, user: usize, item: usize) -> f64 {
        let user_factor = self.user_factors.outer_view(user).unwrap();
        let item_factor = self.item_factors.outer_view(item).unwrap();
        let user_bias = self.user_bias.get(user).unwrap();
        let item_bias = self.item_bias.get(item).unwrap();
        let prediction = self.global_bias + user_bias + item_bias + user_factor.dot(&item_factor);
        prediction
    }
}

fn main() {
    let ratings = vec![
        (0, 0, 5.0),
        (0, 1, 4.0),
        (1, 0, 1.0),
        (1, 1, 2.0),
        (2, 0, 4.0),
    ];

    let mut mf = MatrixFactorization::new(
        3,
        2,
        2,
        0.01,
        0.01);
    mf.train(&ratings, 100);

    // Predict rating for user 2 and item 1
    let prediction = mf.predict(2, 1);
    println!("Prediction: {:.2}", prediction);
}