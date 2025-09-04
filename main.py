from BARTphoBEIT import *
from transformers import BartTokenizer

def main():
    """Main training and evaluation function"""
    
    # Configuration
    config = {
        'vision_model': 'google/vit-base-patch16-224-in21k',
        'text_model': 'vinai/phobert-base',  # Vietnamese BERT
        'decoder_model': 'facebook/bart-base',  # We'll use base BART (could use Vietnamese BART if available)
        'hidden_dim': 768,
        'max_length': 128,
        'batch_size': 8,  # Adjust based on your GPU memory
        'learning_rate': 3e-5,
        'weight_decay': 0.01,
        'num_epochs': 20,
        'freeze_encoders': True,  # Start with frozen encoders
        'image_dir': '/home/tgng/coding/BARTphoBEIT_imple/images',  # Update with your image directory path
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"Using device: {config['device']}")
    
    # Load your dataframe here
    df = pd.read_csv('/home/tgng/coding/BARTphoBEIT_imple/text/evaluate_60k_data_balanced.csv')  # Replace with your actual data loading
    # For demo, create a sample dataframe structure
    # sample_data = {
    #     'index': [0, 1, 2],
    #     'image_name': ['image1.jpg', 'image2.jpg', 'image3.jpg'],
    #     'question': ['Màu gì trong hình?', 'Có bao nhiều người?', 'Đây là đồ vật gì?'],
    #     'answers': [['xanh'], ['hai người'], ['chiếc ô tô']]
    # }
    # df = pd.DataFrame(sample_data)

    # Prepare data
    questions = prepare_data_from_dataframe(df)
    
    # Split into train/validation (80/20)
    split_idx = int(0.8 * len(questions))
    train_questions = questions[:split_idx]
    val_questions = questions[split_idx:]
    
    print(f"Train questions: {len(train_questions)}")
    print(f"Validation questions: {len(val_questions)}")
    
    # Initialize tokenizers and feature extractors
    question_tokenizer = AutoTokenizer.from_pretrained(config['text_model'])
    answer_tokenizer = BartTokenizer.from_pretrained(config['decoder_model'])
    feature_extractor = ViTFeatureExtractor.from_pretrained(config['vision_model'])
    
    # Create datasets
    train_dataset = VietnameseVQADataset(
        train_questions, config['image_dir'], question_tokenizer, answer_tokenizer, feature_extractor, config['max_length']
    )
    val_dataset = VietnameseVQADataset(
        val_questions, config['image_dir'], question_tokenizer, answer_tokenizer, feature_extractor, config['max_length']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2
    )
    
    # Initialize model
    model = VietnameseVQAModel(config)
    model = model.to(config['device'])
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Initialize trainer
    trainer = VQATrainer(model, train_loader, val_loader, config['device'], config)
    
    # Train the model
    trainer.train(config['num_epochs'])
    
    print("Training completed!")
    print("Best model saved as 'best_vqa_model.pth'")
    print("Evaluation results saved as 'evaluation_results.json'")

if __name__ == "__main__":
    main()