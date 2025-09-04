from BARTphoBEIT import *
from setup_app import *

# Cấu hình
config = {
    'vision_model': 'google/vit-base-patch16-224-in21k',
    'text_model': 'vinai/phobert-base',  # Vietnamese BERT
    'decoder_model': 'facebook/bart-base',
    'hidden_dim': 768,
    'max_length': 128,
    'batch_size': 4,  # Điều chỉnh theo GPU memory
    'learning_rate': 3e-5,
    'weight_decay': 0.01,
    'num_epochs': 20,
    'freeze_encoders': True,
    'image_dir': './path/to/your/images',  # Đường dẫn đến thư mục chứa ảnh
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# Chuẩn bị dữ liệu
questions = prepare_data(df)

# Chia train/validation
split_idx = int(0.8 * len(questions))
train_questions = questions[:split_idx]
val_questions = questions[split_idx:]