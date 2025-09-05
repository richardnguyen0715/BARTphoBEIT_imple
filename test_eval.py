from evaluation import *
from setup_app import *
import json
from datetime import datetime

# Config phải match với config đã dùng để train
config = {
    'vision_model': 'google/vit-base-patch16-224-in21k',
    'text_model': 'vinai/phobert-large',
    'decoder_model': 'vinai/bartpho-syllable',
    'hidden_dim': 1024,
    'max_length': 128,
    'batch_size': 8,
    'learning_rate': 3e-5,
    'weight_decay': 0.01,
    'num_epochs': 20,
    'freeze_encoders': False,
    'image_dir': '/home/tgng/coding/BARTphoBEIT_imple/images',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # Add improved configuration parameters
    'decoder_lr': 1e-4,
    'encoder_lr': 1e-5,
    'vision_lr': 5e-6,
    'label_smoothing': 0.1,
    'dropout_rate': 0.2,
}

def enhanced_analyze_answer_distribution(predictions, ground_truths):
    """Enhanced analysis with Vietnamese-specific insights"""
    from model_architechture import normalize_vietnamese_answer
    
    print("\n" + "="*60)
    print("ENHANCED ANSWER DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Normalize answers for analysis
    norm_predictions = [normalize_vietnamese_answer(pred) for pred in predictions]
    norm_ground_truths = [normalize_vietnamese_answer(gt) for gt in ground_truths]
    
    # Original analysis
    analyze_answer_distribution(predictions, ground_truths)
    
    # Enhanced Vietnamese-specific analysis
    print("\nVietnamese-specific Analysis:")
    
    # Check for common Vietnamese answer patterns
    common_patterns = {
        'colors': ['đỏ', 'xanh', 'vàng', 'đen', 'trắng', 'nâu'],
        'numbers': ['một', 'hai', 'ba', 'không', 'nhiều', 'ít'],
        'animals': ['chó', 'mèo', 'chim', 'bò', 'gà', 'cá'],
        'actions': ['đang', 'bay', 'chạy', 'ngồi', 'đứng', 'ăn']
    }
    
    for category, words in common_patterns.items():
        pred_count = sum(1 for pred in norm_predictions if any(word in pred for word in words))
        gt_count = sum(1 for gt in norm_ground_truths if any(word in gt for word in words))
        print(f"  {category.capitalize()}: Predicted {pred_count}, Ground truth {gt_count}")


def save_evaluation_results(metrics, predictions, ground_truths, questions, config, model_path):
    """Save evaluation results to files"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save metrics
    metrics_file = f"evaluation_metrics_{timestamp}.json"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump({
            'metrics': metrics,
            'config': config,
            'model_path': model_path,
            'timestamp': timestamp,
            'total_samples': len(predictions)
        }, f, indent=2, ensure_ascii=False)
    
    # Save detailed results
    results_file = f"evaluation_results_{timestamp}.json"
    detailed_results = []
    for i, (q, p, gt) in enumerate(zip(questions[:100], predictions[:100], ground_truths[:100])):  # Save first 100 samples
        detailed_results.append({
            'id': i,
            'question': q,
            'prediction': p,
            'ground_truth': gt,
            'exact_match': p.strip().lower() == gt.strip().lower()
        })
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to:")
    print(f"  Metrics: {metrics_file}")
    print(f"  Detailed results: {results_file}")

def compare_models():
    """Compare finetuned model vs base model"""
    
    print("Loading and preparing test data...")
    df = pd.read_csv('/home/tgng/coding/BARTphoBEIT_imple/text/evaluate_60k_data_balanced.csv')
    questions = prepare_data_from_dataframe(df)
    
    # Use a smaller subset for faster evaluation during testing
    split_idx = int(0.8 * len(questions))
    test_questions = questions[split_idx:]  # Use 500 samples for quick evaluation: [split_idx:split_idx + 500]
    
    print(f"Total test samples: {len(test_questions)}")
    
    results = {}
    
    # Evaluate finetuned model
    print("\n" + "="*80)
    print("EVALUATING FINETUNED MODEL")
    print("="*80)
    
    finetuned_metrics, finetuned_predictions, finetuned_ground_truths = evaluate_model(
        model_path="/home/tgng/coding/BARTphoBEIT_imple/best_vqa_model.pth",
        test_questions=test_questions, 
        config=config,
        load_pretrained=True
    )
    
    results['finetuned'] = finetuned_metrics
    
    # Analyze answer distribution for finetuned model
    # analyze_answer_distribution(finetuned_predictions, finetuned_ground_truths)
    enhanced_analyze_answer_distribution(finetuned_predictions, finetuned_ground_truths)
    
    # Save results
    save_evaluation_results(
        finetuned_metrics, finetuned_predictions, finetuned_ground_truths, 
        [q['question'] for q in test_questions], config, 
        "/home/tgng/coding/BARTphoBEIT_imple/best_vqa_model.pth"
    )
    
    # Evaluate base model for comparison
    print("\n" + "="*80)
    print("EVALUATING BASE MODEL (NO FINETUNING)")
    print("="*80)
    
    base_metrics, base_predictions, base_ground_truths = evaluate_model(
        model_path=None,
        test_questions=test_questions, 
        config=config,
        load_pretrained=False
    )
    
    results['base'] = base_metrics
    
    # Compare results
    print("\n" + "="*100)
    print("MODEL COMPARISON")
    print("="*100)
    
    comparison_metrics = [
        'exact_match', 'accuracy', 'precision', 'recall', 'f1', 
        'bleu_1', 'bleu_2', 'bleu_3', 'bleu_4', 
        'meteor', 'rouge', 'cider'
    ]
    
    print(f"{'Metric':<15} {'Finetuned':<12} {'Base':<12} {'Improvement':<12}")
    print("-" * 55)
    
    for metric in comparison_metrics:
        finetuned_score = finetuned_metrics.get(metric, 0)
        base_score = base_metrics.get(metric, 0)
        improvement = finetuned_score - base_score
        improvement_pct = (improvement / base_score * 100) if base_score > 0 else 0
        
        print(f"{metric:<15} {finetuned_score:<12.4f} {base_score:<12.4f} {improvement:+.4f} ({improvement_pct:+.1f}%)")
    
    print("="*100)
    
    # Identify areas of improvement and degradation
    print("\nKEY FINDINGS:")
    
    significant_improvements = []
    significant_degradations = []
    
    for metric in comparison_metrics:
        finetuned_score = finetuned_metrics.get(metric, 0)
        base_score = base_metrics.get(metric, 0)
        
        if base_score > 0:
            improvement_pct = (finetuned_score - base_score) / base_score * 100
            
            if improvement_pct > 5:  # More than 5% improvement
                significant_improvements.append((metric, improvement_pct))
            elif improvement_pct < -5:  # More than 5% degradation
                significant_degradations.append((metric, improvement_pct))
    
    if significant_improvements:
        print("✓ Significant improvements:")
        for metric, improvement in significant_improvements:
            print(f"  - {metric}: {improvement:+.1f}%")
    
    if significant_degradations:
        print("✗ Areas needing attention:")
        for metric, degradation in significant_degradations:
            print(f"  - {metric}: {degradation:+.1f}%")
    
    if not significant_improvements and not significant_degradations:
        print("→ Performance is similar between models")
    
    return results

if __name__ == "__main__":
    try:
        print("Starting comprehensive VQA evaluation...")
        print(f"Using device: {config['device']}")
        print(f"Batch size: {config['batch_size']}")
        
        results = compare_models()
        
        print("\nEvaluation completed successfully!")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()