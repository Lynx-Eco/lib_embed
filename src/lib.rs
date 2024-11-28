use anyhow::Result;
use ndarray::{ Array1, Array2 };
use ort::{
    environment::Environment,
    session::Session,
    session::builder::GraphOptimizationLevel,
    value::Value,
};
use std::{ path::Path, sync::Arc };

pub struct OnnxModel {
    session: Session,
    environment: Arc<Environment>,
}

impl OnnxModel {
    /// Create a new ONNX model from a file path
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        // Create an environment for the ONNX Runtime
        let environment = ort::init().with_name("onnx_model_environment").commit()?;

        // Create a session for running the model
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;

        Ok(Self {
            session,
            environment,
        })
    }

    /// Run inference on the model with the given input
    pub fn run(
        &self,
        input_ids: Array2<i64>,
        attention_mask: Array2<i64>,
        token_type_ids: Array2<i64>
    ) -> Result<Array2<f32>> {
        // Create input tensors
        let input_ids_tensor = Value::from_array((
            input_ids.shape(),
            input_ids.as_slice().unwrap(),
        ))?;
        let attention_mask_tensor = Value::from_array((
            attention_mask.shape(),
            attention_mask.as_slice().unwrap(),
        ))?;
        let token_type_ids_tensor = Value::from_array((
            token_type_ids.shape(),
            token_type_ids.as_slice().unwrap(),
        ))?;

        // Run inference with all inputs
        let outputs = self.session.run(
            (ort::inputs! {
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
                "token_type_ids" => token_type_ids_tensor
            })?
        )?;

        // Get first output tensor
        let values: Vec<_> = outputs.values().collect();
        let output = values[0].try_extract_tensor::<f32>()?;
        let array_view = output.view();
        let shape = array_view.shape();
        let data: Vec<f32> = array_view.as_slice().unwrap().to_vec();
        let output_array = Array2::from_shape_vec((shape[0], shape[1]), data)?;

        Ok(output_array)
    }

    /// Normalize vectors (L2 normalization)
    pub fn normalize(&self, vectors: Array2<f32>) -> Array2<f32> {
        let mut normalized = Array2::zeros(vectors.raw_dim());
        for (i, row) in vectors.outer_iter().enumerate() {
            let norm = row.dot(&row).sqrt();
            let normalized_row: Array1<f32> = if norm > 0.0 {
                row.map(|&x| x / norm)
            } else {
                Array1::zeros(row.len())
            };
            normalized.row_mut(i).assign(&normalized_row);
        }
        normalized
    }

    /// Calculate similarities between queries and documents
    pub fn calculate_similarities(
        &self,
        query_vectors: Array2<f32>,
        doc_vectors: Array2<f32>
    ) -> Array2<f32> {
        query_vectors.dot(&doc_vectors.t())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_model_loading() {
        let api = hf_hub::api::sync::Api::new().unwrap();
        let repo = api.repo(
            hf_hub::Repo::with_revision(
                "dunzhang/stella_en_400M_v5".to_string(),
                hf_hub::RepoType::Model,
                "refs/pr/3".to_string()
            )
        );
        let model_path = repo.get("onnx/model.onnx").unwrap();

        let result = OnnxModel::new(model_path);
        if let Err(e) = &result {
            println!("Error loading model: {}", e);
        }
        assert!(result.is_ok());
    }

    #[test]
    fn test_normalization() {
        let api = hf_hub::api::sync::Api::new().unwrap();
        let repo = api.repo(
            hf_hub::Repo::with_revision(
                "dunzhang/stella_en_400M_v5".to_string(),
                hf_hub::RepoType::Model,
                "refs/pr/3".to_string()
            )
        );
        let model_path = repo.get("onnx/model.onnx").unwrap();

        let model = match OnnxModel::new(model_path) {
            Ok(m) => m,
            Err(e) => {
                println!("Error loading model: {}", e);
                panic!("Failed to load model");
            }
        };
        let vectors = arr2(
            &[
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        );

        let normalized = model.normalize(vectors);

        // Check that each row has unit norm
        for row in normalized.outer_iter() {
            let norm = row.dot(&row).sqrt();
            assert!((norm - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_similarities() {
        let api = hf_hub::api::sync::Api::new().unwrap();
        let repo = api.repo(
            hf_hub::Repo::with_revision(
                "dunzhang/stella_en_400M_v5".to_string(),
                hf_hub::RepoType::Model,
                "refs/pr/3".to_string()
            )
        );
        let model_path = repo.get("onnx/model.onnx").unwrap();

        let model = match OnnxModel::new(model_path) {
            Ok(m) => m,
            Err(e) => {
                println!("Error loading model: {}", e);
                panic!("Failed to load model");
            }
        };
        let query_vectors = arr2(
            &[
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        );
        let doc_vectors = arr2(
            &[
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        );

        let similarities = model.calculate_similarities(query_vectors, doc_vectors);

        assert_eq!(similarities.shape(), &[2, 2]);
        assert!((similarities[[0, 0]] - 1.0).abs() < 1e-6);
        assert!((similarities[[1, 1]] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_sentence_embedding() {
        // Download tokenizer and model from HF Hub
        let api = hf_hub::api::sync::Api::new().unwrap();
        let repo = api.repo(
            hf_hub::Repo::with_revision(
                "dunzhang/stella_en_400M_v5".to_string(),
                hf_hub::RepoType::Model,
                "refs/pr/3".to_string()
            )
        );

        let tokenizer_path = repo.get("tokenizer.json").unwrap();
        let model_path = repo.get("onnx/model.onnx").unwrap();

        // Load tokenizer
        let mut tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path).unwrap();
        tokenizer.with_padding(
            Some(tokenizers::PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                direction: tokenizers::PaddingDirection::Right,
                ..Default::default()
            })
        );

        // Load model from downloaded path
        let model = match OnnxModel::new(model_path) {
            Ok(m) => m,
            Err(e) => {
                println!("Error loading model: {}", e);
                panic!("Failed to load model");
            }
        };

        // Example queries and documents
        let queries = [
            "What are some ways to reduce stress?",
            "What are the benefits of drinking green tea?",
        ];

        let docs = [
            "There are many effective ways to reduce stress. Some common techniques include deep breathing, meditation, and physical activity. Engaging in hobbies, spending time in nature, and connecting with loved ones can also help alleviate stress. Additionally, setting boundaries, practicing self-care, and learning to say no can prevent stress from building up.",
            "Green tea has been consumed for centuries and is known for its potential health benefits. It contains antioxidants that may help protect the body against damage caused by free radicals. Regular consumption of green tea has been associated with improved heart health, enhanced cognitive function, and a reduced risk of certain types of cancer. The polyphenols in green tea may also have anti-inflammatory and weight loss properties.",
        ];

        // Preprocess queries with instruction
        let query_texts: Vec<String> = queries
            .iter()
            .map(|q| {
                format!("Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery: {}", q)
            })
            .collect();

        // Tokenize and encode queries
        let query_encodings = tokenizer.encode_batch(query_texts, true).unwrap();

        // Create input tensors for queries
        let query_input_ids = Array2::from_shape_vec(
            (queries.len(), query_encodings[0].get_ids().len()),
            query_encodings
                .iter()
                .flat_map(|e|
                    e
                        .get_ids()
                        .iter()
                        .map(|&x| x as i64)
                )
                .collect()
        ).unwrap();

        let query_attention_mask = Array2::from_shape_vec(
            (queries.len(), query_encodings[0].get_attention_mask().len()),
            query_encodings
                .iter()
                .flat_map(|e|
                    e
                        .get_attention_mask()
                        .iter()
                        .map(|&x| x as i64)
                )
                .collect()
        ).unwrap();

        // Create token_type_ids (all zeros for single sequence)
        let query_token_type_ids = Array2::zeros(query_input_ids.raw_dim());

        // Tokenize and encode documents (no instruction prefix)
        let doc_encodings = tokenizer.encode_batch(docs.to_vec(), true).unwrap();

        // Create input tensors for documents
        let doc_input_ids = Array2::from_shape_vec(
            (docs.len(), doc_encodings[0].get_ids().len()),
            doc_encodings
                .iter()
                .flat_map(|e|
                    e
                        .get_ids()
                        .iter()
                        .map(|&x| x as i64)
                )
                .collect()
        ).unwrap();

        let doc_attention_mask = Array2::from_shape_vec(
            (docs.len(), doc_encodings[0].get_attention_mask().len()),
            doc_encodings
                .iter()
                .flat_map(|e|
                    e
                        .get_attention_mask()
                        .iter()
                        .map(|&x| x as i64)
                )
                .collect()
        ).unwrap();

        // Create token_type_ids (all zeros for single sequence)
        let doc_token_type_ids = Array2::zeros(doc_input_ids.raw_dim());

        // Get embeddings for queries and documents
        let query_embeddings = model
            .run(query_input_ids, query_attention_mask, query_token_type_ids)
            .unwrap();
        let doc_embeddings = model
            .run(doc_input_ids, doc_attention_mask, doc_token_type_ids)
            .unwrap();

        // Normalize embeddings
        let query_embeddings = model.normalize(query_embeddings);
        let doc_embeddings = model.normalize(doc_embeddings);

        // Calculate similarities
        let similarities = model.calculate_similarities(query_embeddings, doc_embeddings);

        println!("Similarity matrix:");
        println!("{:?}", similarities);

        // Check dimensions
        assert_eq!(similarities.shape(), &[2, 2]);

        // The diagonal should have higher values (queries match their corresponding documents)
        assert!(similarities[[0, 0]] > similarities[[0, 1]]);
        assert!(similarities[[1, 1]] > similarities[[1, 0]]);
    }
}
