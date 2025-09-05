from BARTphoBEIT import *

def fine_tune_model(pretrained_path, train_questions, val_questions, config):
    """Fine-tune từ model đã train"""
    
    # Load pretrained model
    model = VietnameseVQAModel(config)
    model.load_state_dict(torch.load(pretrained_path))
    
    # Unfreeze some layers for fine-tuning
    config['freeze_encoders'] = False  # Unfreeze for fine-tuning
    config['learning_rate'] = 1e-5  # Lower learning rate for fine-tuning
    config['num_epochs'] = 5  # Fewer epochs
    
    # Create data loaders
    tokenizer = AutoTokenizer.from_pretrained(config['text_model'])
    feature_extractor = ViTFeatureExtractor.from_pretrained(config['vision_model'])
    
    train_dataset = VietnameseVQADataset(train_questions, config['image_dir'], tokenizer, feature_extractor)
    val_dataset = VietnameseVQADataset(val_questions, config['image_dir'], tokenizer, feature_extractor)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Fine-tune
    trainer = VQATrainer(model, train_loader, val_loader, config['device'], config)
    trainer.train(config['num_epochs'])

# 7. PHÂN TÍCH KẾT QUẢ CHI TIẾT
def analyze_results(predictions, ground_truths, questions):
    """Phân tích chi tiết kết quả evaluation"""
    
    results_analysis = {
        'correct_predictions': [],
        'wrong_predictions': [],
        'question_types': {},
        'error_analysis': {}
    }
    
    for i, (pred, gt, q) in enumerate(zip(predictions, ground_truths, questions)):
        pred_norm = pred.lower().strip()
        gt_norm = gt.lower().strip()
        
        result = {
            'question_id': q.get('question_id', i),
            'question': q['question'],
            'predicted': pred,
            'ground_truth': gt,
            'correct': pred_norm == gt_norm
        }
        
        if result['correct']:
            results_analysis['correct_predictions'].append(result)
        else:
            results_analysis['wrong_predictions'].append(result)
        
        # Phân loại theo loại câu hỏi
        question_type = classify_question_type(q['question'])
        if question_type not in results_analysis['question_types']:
            results_analysis['question_types'][question_type] = {'correct': 0, 'total': 0}
        results_analysis['question_types'][question_type]['total'] += 1
        if result['correct']:
            results_analysis['question_types'][question_type]['correct'] += 1
    
    # In kết quả phân tích
    print("\n=== PHÂN TÍCH KẾT QUẢ ===")
    print(f"Tổng số câu hỏi: {len(predictions)}")
    print(f"Trả lời đúng: {len(results_analysis['correct_predictions'])}")
    print(f"Trả lời sai: {len(results_analysis['wrong_predictions'])}")
    
    print("\n=== THEO LOẠI CÂU HỎI ===")
    for q_type, stats in results_analysis['question_types'].items():
        accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"{q_type}: {stats['correct']}/{stats['total']} ({accuracy:.2%})")
    
    print("\n=== MỘT SỐ LỖI THƯỜNG GẶP ===")
    for i, error in enumerate(results_analysis['wrong_predictions'][:5]):  # Show first 5 errors
        print(f"\nLỗi {i+1}:")
        print(f"  Câu hỏi: {error['question']}")
        print(f"  Dự đoán: {error['predicted']}")
        print(f"  Đáp án đúng: {error['ground_truth']}")
    
    return results_analysis

def classify_question_type(question):
    """Phân loại loại câu hỏi"""
    question_lower = question.lower()
    
    if any(word in question_lower for word in ['màu', 'color']):
        return 'Color'
    elif any(word in question_lower for word in ['bao nhiêu', 'mấy', 'số']):
        return 'Number'
    elif any(word in question_lower for word in ['ở đâu', 'where', 'vị trí']):
        return 'Location'
    elif any(word in question_lower for word in ['là gì', 'what', 'cái gì']):
        return 'Object'
    else:
        return 'Other'