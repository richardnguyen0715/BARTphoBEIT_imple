from BARTphoBEIT import *

def evaluate_model(model_path, test_questions, config, load_pretrained=True):
    """Evaluate trained model"""
    
    # Load model
    model = VietnameseVQAModel(config)
    
    # Only load pretrained weights if specified
    if load_pretrained and model_path and os.path.exists(model_path):
        print(f"Loading finetuned model from {model_path}")
        model.load_state_dict(torch.load(model_path))
    else:
        print("Using base model (no finetuned weights loaded)")
    
    model = model.to(config['device'])
    model.eval()
    
    # Prepare test dataset
    question_tokenizer = AutoTokenizer.from_pretrained(config['text_model'])
    answer_tokenizer = BartTokenizer.from_pretrained(config['decoder_model'])
    feature_extractor = ViTFeatureExtractor.from_pretrained(config['vision_model'])
    
    test_dataset = VietnameseVQADataset(
        test_questions, config['image_dir'], question_tokenizer, answer_tokenizer, feature_extractor
    )
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Evaluate
    evaluator = VQAEvaluator(model.decoder_tokenizer)
    predictions = []
    ground_truths = []
    sample_questions = []  # Store questions for sample display
    
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
            sample_questions.extend(batch['question_text'])
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics(predictions, ground_truths)
    
    print("=== EVALUATION RESULTS ===")
    print(f"Model type: {'Finetuned' if load_pretrained and model_path and os.path.exists(model_path) else 'Base (no finetuning)'}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Exact matches: {metrics['exact_match_count']}/{metrics['total_count']}")
    
    # Display sample results
    print_sample_results(sample_questions, predictions, ground_truths)
    
    return metrics, predictions, ground_truths

def print_sample_results(questions, predictions, ground_truths, num_samples=10):
    """Print sample results from the model"""
    print("\n=== SAMPLE RESULTS ===")
    print("-" * 80)
    
    # Show first few samples
    for i in range(min(num_samples, len(questions))):
        print(f"Sample {i+1}:")
        print(f"Question: {questions[i]}")
        print(f"Predicted: {predictions[i]}")
        print(f"Ground Truth: {ground_truths[i]}")
        
        # Check if prediction matches ground truth
        is_correct = predictions[i].strip().lower() == ground_truths[i].strip().lower()
        print(f"Result: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
        print("-" * 40)
    
    print(f"Showing {min(num_samples, len(questions))} out of {len(questions)} total samples")
    print("-" * 80)