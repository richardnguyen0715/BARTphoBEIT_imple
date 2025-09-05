from BARTphoBEIT import *
from transformers import AutoTokenizer, ViTFeatureExtractor
try:
    from transformers import BartphoTokenizer
except ImportError:
    # Fallback nếu không có BartphoTokenizer
    from transformers import AutoTokenizer as BartphoTokenizer

def main():
    """Main training and evaluation function"""
    config = {
        'vision_model': 'google/vit-base-patch16-224-in21k', # hoặc sài Salesforce/blip2-opt-2.7b, openai/clip-vit-base-patch32
        'text_model': 'vinai/phobert-large',  
        'decoder_model': 'vinai/bartpho-syllable',  # MBart model # Với VQA, thường dữ liệu ngắn → syllable sẽ robust hơn bản word!!!.
        'hidden_dim': 1024,  # PhoBERT-large dimension
        'max_length': 128,
        'batch_size': 4,  # Giảm batch size do model khá là lớn
        'learning_rate': 2e-5,
        'weight_decay': 0.01,
        'num_epochs': 20,
        'freeze_encoders': True,
        'image_dir': '/home/tgng/coding/BARTphoBEIT_imple/images',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"Using device: {config['device']}")
    print("Loading data...")
    
    # Load dataframe
    df = pd.read_csv('/home/tgng/coding/BARTphoBEIT_imple/text/evaluate_60k_data_balanced.csv')

    # Prepare data
    questions = prepare_data_from_dataframe(df)
    
    # Split into train/validation (80/20)
    split_idx = int(0.8 * len(questions))
    train_questions = questions[:split_idx]
    val_questions = questions[split_idx:]
    
    print(f"Train questions: {len(train_questions)}")
    print(f"Validation questions: {len(val_questions)}")
    
    # Initialize tokenizers and feature extractors
    print("Loading tokenizers and feature extractor...")
    question_tokenizer = AutoTokenizer.from_pretrained(config['text_model'])
    answer_tokenizer = AutoTokenizer.from_pretrained(config['decoder_model'])
    feature_extractor = ViTFeatureExtractor.from_pretrained(config['vision_model'])
    
    # Test tokenizer để đảm bảo hoạt động đúng
    print("Testing tokenizers...")
    test_question = "Con gì đang bay trong bức tranh?"
    test_answer = "Con chim"
    
    q_tokens = question_tokenizer(test_question)
    a_tokens = answer_tokenizer(test_answer)
    print(f"Question tokens: {len(q_tokens['input_ids'])}")
    print(f"Answer tokens: {len(a_tokens['input_ids'])}")
    
    print("Creating datasets...")
    # Create datasets
    train_dataset = VietnameseVQADataset(
        train_questions, config['image_dir'], question_tokenizer, answer_tokenizer, feature_extractor, config['max_length']
    )
    val_dataset = VietnameseVQADataset(
        val_questions, config['image_dir'], question_tokenizer, answer_tokenizer, feature_extractor, config['max_length']
    )
    
    # Create data loaders  
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0  # num_workers=0 để tránh lỗi
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0
    )
    
    print("Initializing model...")
    # Initialize model
    try:
        model = VietnameseVQAModel(config)
        model = model.to(config['device'])
    except Exception as e:
        print(f"Error initializing model: {e}")
        return
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model initialized with {total_params:,} parameters")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    
    # Test một batch trước khi training
    print("Testing data loading...")
    try:
        test_batch = next(iter(train_loader))
        print(f"Batch keys: {test_batch.keys()}")
        for key, value in test_batch.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: {value.shape}")
    except Exception as e:
        print(f"Error in data loading: {e}")
        return
    
    # Initialize trainer
    trainer = VQATrainer(model, train_loader, val_loader, config['device'], config)
    
    print("Starting training...")
    # Train the model
    try:
        trainer.train(config['num_epochs'])
    except Exception as e:
        print(f"Error during training: {e}")
        return
    
    print("Training completed!")
    print("Best model saved as 'best_vqa_model.pth'")
    print("Evaluation results saved as 'evaluation_results.json'")

if __name__ == "__main__":
    main()