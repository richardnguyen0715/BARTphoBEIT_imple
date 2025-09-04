from BARTphoBEIT import *

def evaluate_model(model_path, test_questions, config):
    """Evaluate trained model"""
    
    # Load model
    model = VietnameseVQAModel(config)
    model.load_state_dict(torch.load(model_path))
    model = model.to(config['device'])
    model.eval()
    
    # Prepare test dataset
    tokenizer = AutoTokenizer.from_pretrained(config['text_model'])
    feature_extractor = ViTFeatureExtractor.from_pretrained(config['vision_model'])
    
    test_dataset = VietnameseVQADataset(
        test_questions, config['image_dir'], tokenizer, feature_extractor
    )
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Evaluate
    evaluator = VQAEvaluator(model.decoder_tokenizer)
    predictions = []
    ground_truths = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(config['device'])
            
            generated_ids = model(
                pixel_values=batch['pixel_values'],
                question_input_ids=batch['question_input_ids'],
                question_attention_mask=batch['question_attention_mask']
            )
            
            pred_texts = model.decoder_tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            
            predictions.extend(pred_texts)
            ground_truths.extend(batch['answer_text'])
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics(predictions, ground_truths)
    
    print("=== EVALUATION RESULTS ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Exact matches: {metrics['exact_match_count']}/{metrics['total_count']}")
    
    return metrics, predictions, ground_truths