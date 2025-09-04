from evaluation import *
from fintune import *
from setup_app import *

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
    'image_dir': '/home/tgng/coding/BARTphoBEIT_imple/images',  # Đường dẫn đến thư mục chứa ảnh
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

print("Loading and preparing test data...")
df = pd.read_csv('/home/tgng/coding/BARTphoBEIT_imple/text/evaluate_60k_data_balanced.csv')
questions = prepare_data(df)
split_idx = int(0.8 * len(questions))
test_questions = questions[split_idx:]  # Use test set

print(f"Total test samples: {len(test_questions)}")

print("\nEvaluating model...")
metrics, predictions, ground_truths = evaluate_model(
    'best_vqa_model.pth', test_questions, config
)

print("\nAnalyzing results...")
# 5. Analysis
results_analysis = analyze_results(predictions, ground_truths, test_questions)

# Show some statistics about answer lengths
print("\n=== ANSWER STATISTICS ===")
pred_lengths = [len(pred.split()) for pred in predictions]
truth_lengths = [len(truth.split()) for truth in ground_truths]

print(f"Average predicted answer length: {sum(pred_lengths)/len(pred_lengths):.2f} words")
print(f"Average ground truth answer length: {sum(truth_lengths)/len(truth_lengths):.2f} words")
print(f"Max predicted answer length: {max(pred_lengths)} words")
print(f"Max ground truth answer length: {max(truth_lengths)} words")