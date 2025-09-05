import torch
import torch.nn as nn
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm
import os
import json
import glob
from collections import defaultdict
import wandb
from datetime import datetime
from model_architechture import ImprovedVietnameseVQAModel, normalize_vietnamese_answer

# Install required packages for evaluation
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    import nltk
    nltk.download('punkt', quiet=True)
    BLEU_ROUGE_AVAILABLE = True
except ImportError:
    print("NLTK/Rouge not available. Install with: pip install nltk rouge-score")
    BLEU_ROUGE_AVAILABLE = False

class ImprovedVQATrainer:
    """Enhanced trainer with all improvements"""
    
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        self.current_stage = 1
        self.global_step = 0
        self.setup_optimizers_and_schedulers()
        
        # For evaluation
        self.best_f1 = 0
        self.best_accuracy = 0
        self.best_fuzzy_accuracy = 0
        
        # Setup logging
        self.setup_logging()
        
        # Checkpoint management
        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Evaluation metrics
        if BLEU_ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
            self.smoothing_function = SmoothingFunction().method4
    
    def setup_logging(self):
        """Setup wandb logging"""
        if self.config.get('use_wandb', False):
            try:
                wandb.init(
                    project=self.config.get('project_name', 'Vietnamese-VQA'),
                    config=self.config,
                    name=f"VQA_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                self.use_wandb = True
                print("Wandb logging initialized")
            except Exception as e:
                print(f"Failed to initialize wandb: {e}")
                self.use_wandb = False
        else:
            self.use_wandb = False
    
    def setup_optimizers_and_schedulers(self):
        """Setup optimizers with improved scheduler"""
        
        # Group parameters by component
        decoder_params = []
        encoder_params = []
        vision_params = []
        fusion_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
                
            if 'text_decoder' in name:
                decoder_params.append(param)
            elif 'text_model' in name:
                encoder_params.append(param)
            elif 'vision_model' in name:
                vision_params.append(param)
            else:  # fusion layer and projections
                fusion_params.append(param)
        
        # Setup parameter groups with different learning rates
        param_groups = []
        
        if decoder_params:
            param_groups.append({
                'params': decoder_params, 
                'lr': self.config['decoder_lr'],
                'name': 'decoder'
            })
        
        if encoder_params:
            param_groups.append({
                'params': encoder_params, 
                'lr': self.config['encoder_lr'],
                'name': 'encoder'
            })
        
        if vision_params:
            param_groups.append({
                'params': vision_params, 
                'lr': self.config['vision_lr'],
                'name': 'vision'
            })
        
        if fusion_params:
            param_groups.append({
                'params': fusion_params, 
                'lr': self.config['decoder_lr'],  # Same as decoder
                'name': 'fusion'
            })
        
        # Create optimizer
        self.optimizer = optim.AdamW(
            param_groups,
            weight_decay=self.config['weight_decay']
        )
        
        # Setup improved scheduler with warmup + linear decay
        total_steps = len(self.train_loader) * self.config['num_epochs']
        warmup_steps = int(total_steps * self.config.get('warmup_ratio', 0.1))
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        print(f"Enhanced optimizer setup:")
        print(f"  Parameter groups: {len(param_groups)}")
        print(f"  Total training steps: {total_steps:,}")
        print(f"  Warmup steps: {warmup_steps:,}")
        
        for group in param_groups:
            print(f"  {group['name']}: {len(group['params'])} params, lr={group['lr']}")
    
    def train_epoch(self, epoch):
        """Enhanced training epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        # Stage management
        if epoch == self.config['stage1_epochs']:
            print("\n" + "="*60)
            print("SWITCHING TO STAGE 2: Partial encoder unfreezing")
            print("="*60)
            self.model.partial_unfreeze(self.config['unfreeze_last_n_layers'])
            self.setup_optimizers_and_schedulers()  # Recreate optimizer
            self.current_stage = 2
        
        progress_bar = tqdm(self.train_loader, desc=f"Stage {self.current_stage} - Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(
                pixel_values=batch['pixel_values'],
                question_input_ids=batch['question_input_ids'],
                question_attention_mask=batch['question_attention_mask'],
                answer_input_ids=batch['answer_input_ids'],
                answer_attention_mask=batch['answer_attention_mask']
            )
            
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            self.global_step += 1
            
            # Update progress bar
            current_lr = self.scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Avg Loss': f"{total_loss/(batch_idx+1):.4f}",
                'LR': f"{current_lr:.2e}",
                'Stage': self.current_stage
            })
            
            # Log to wandb
            if self.use_wandb and batch_idx % 50 == 0:
                wandb.log({
                    'train_loss': loss.item(),
                    'learning_rate': current_lr,
                    'epoch': epoch,
                    'stage': self.current_stage,
                    'global_step': self.global_step
                })
            
            # Evaluate every N steps
            if (self.config.get('evaluate_every_n_steps', 1000) > 0 and 
                self.global_step % self.config['evaluate_every_n_steps'] == 0):
                
                print(f"\nEvaluating at step {self.global_step}...")
                val_metrics, predictions, ground_truths = self.evaluate_vqa()
                
                if self.use_wandb:
                    wandb.log({
                        'val_accuracy': val_metrics['accuracy'],
                        'val_fuzzy_accuracy': val_metrics['fuzzy_accuracy'],
                        'val_f1_score': val_metrics['f1_score'],
                        'val_bleu_1': val_metrics.get('bleu_1', 0),
                        'val_rouge_l': val_metrics.get('rouge_l', 0),
                        'global_step': self.global_step
                    })
                
                self.model.train()  # Back to training mode
        
        return total_loss / num_batches
    
    def evaluate_vqa(self):
        """Enhanced VQA evaluation with comprehensive metrics"""
        self.model.eval()
        predictions = []
        ground_truths = []
        
        progress_bar = tqdm(self.val_loader, desc="Evaluating", leave=False)
        
        with torch.no_grad():
            for batch in progress_bar:
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                # Generate predictions
                generated_ids = self.model(
                    pixel_values=batch['pixel_values'],
                    question_input_ids=batch['question_input_ids'],
                    question_attention_mask=batch['question_attention_mask']
                )
                
                # Decode predictions
                pred_texts = self.model.decoder_tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
                
                predictions.extend(pred_texts)
                ground_truths.extend(batch['answer_text'])
        
        # Calculate comprehensive metrics
        metrics = self.calculate_comprehensive_metrics(predictions, ground_truths)
        
        return metrics, predictions, ground_truths
    
    def calculate_comprehensive_metrics(self, predictions, ground_truths):
        """Calculate comprehensive VQA metrics including BLEU, ROUGE, CIDEr"""
        from model_architechture import normalize_vietnamese_answer
        
        # Normalize answers
        norm_predictions = [normalize_vietnamese_answer(pred) for pred in predictions]
        norm_ground_truths = [normalize_vietnamese_answer(gt) for gt in ground_truths]
        
        metrics = {}
        
        # Basic accuracy metrics
        exact_matches = [pred == gt for pred, gt in zip(norm_predictions, norm_ground_truths)]
        metrics['accuracy'] = np.mean(exact_matches)
        
        # Fuzzy matching with improved scoring
        fuzzy_matches = []
        for pred, gt in zip(norm_predictions, norm_ground_truths):
            if pred == gt:
                fuzzy_matches.append(1.0)
            elif pred in gt or gt in pred:
                fuzzy_matches.append(0.8)  # Partial credit
            else:
                # Check word overlap
                pred_words = set(pred.split())
                gt_words = set(gt.split())
                if pred_words and gt_words:
                    overlap = len(pred_words.intersection(gt_words))
                    total = len(pred_words.union(gt_words))
                    fuzzy_matches.append(overlap / total if total > 0 else 0.0)
                else:
                    fuzzy_matches.append(0.0)
        
        metrics['fuzzy_accuracy'] = np.mean(fuzzy_matches)
        
        # Token-level F1 score
        f1_scores = []
        precisions = []
        recalls = []
        
        for pred, gt in zip(norm_predictions, norm_ground_truths):
            pred_tokens = set(pred.split())
            gt_tokens = set(gt.split())
            
            if len(pred_tokens) == 0 and len(gt_tokens) == 0:
                f1_scores.append(1.0)
                precisions.append(1.0)
                recalls.append(1.0)
            elif len(pred_tokens) == 0:
                f1_scores.append(0.0)
                precisions.append(0.0)
                recalls.append(0.0)
            else:
                common_tokens = pred_tokens.intersection(gt_tokens)
                precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
                recall = len(common_tokens) / len(gt_tokens) if gt_tokens else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1)
        
        metrics['precision'] = np.mean(precisions)
        metrics['recall'] = np.mean(recalls)
        metrics['f1_score'] = np.mean(f1_scores)
        
        # BLEU and ROUGE scores
        if BLEU_ROUGE_AVAILABLE and self.config.get('calculate_bleu_rouge', True):
            bleu_scores = {'bleu_1': [], 'bleu_2': [], 'bleu_3': [], 'bleu_4': []}
            rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
            
            for pred, gt in zip(norm_predictions, norm_ground_truths):
                pred_tokens = pred.split()
                gt_tokens = gt.split()
                
                # BLEU scores
                for n in range(1, 5):
                    try:
                        bleu = sentence_bleu([gt_tokens], pred_tokens, 
                                           weights=tuple([1/n]*n + [0]*(4-n)),
                                           smoothing_function=self.smoothing_function)
                        bleu_scores[f'bleu_{n}'].append(bleu)
                    except:
                        bleu_scores[f'bleu_{n}'].append(0.0)
                
                # ROUGE scores
                try:
                    rouge_result = self.rouge_scorer.score(gt, pred)
                    rouge_scores['rouge1'].append(rouge_result['rouge1'].fmeasure)
                    rouge_scores['rouge2'].append(rouge_result['rouge2'].fmeasure)
                    rouge_scores['rougeL'].append(rouge_result['rougeL'].fmeasure)
                except:
                    rouge_scores['rouge1'].append(0.0)
                    rouge_scores['rouge2'].append(0.0)
                    rouge_scores['rougeL'].append(0.0)
            
            # Add averaged scores to metrics
            for key, scores in bleu_scores.items():
                metrics[key] = np.mean(scores) if scores else 0.0
            
            for key, scores in rouge_scores.items():
                metrics[key.lower()] = np.mean(scores) if scores else 0.0
        
        # Add counts
        metrics['exact_match_count'] = sum(exact_matches)
        metrics['total_count'] = len(predictions)
        
        return metrics
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Enhanced checkpoint saving"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'metrics': metrics,
            'current_stage': self.current_stage
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = 'best_vqa_model.pth'
            torch.save(checkpoint, best_path)
            print(f"✓ New best model saved: {best_path}")
        
        # Keep only last N checkpoints
        self.cleanup_checkpoints()
        
        return checkpoint_path
    
    def cleanup_checkpoints(self):
        """Keep only the last N checkpoints"""
        keep_n = self.config.get('keep_last_n_checkpoints', 5)
        checkpoints = glob.glob(os.path.join(self.checkpoint_dir, 'checkpoint_epoch_*.pth'))
        
        if len(checkpoints) > keep_n:
            # Sort by creation time
            checkpoints.sort(key=os.path.getctime)
            
            # Remove oldest checkpoints
            for checkpoint in checkpoints[:-keep_n]:
                try:
                    os.remove(checkpoint)
                    print(f"Removed old checkpoint: {os.path.basename(checkpoint)}")
                except:
                    pass
    
    def save_predictions(self, predictions, ground_truths, epoch, metrics):
        """Save predictions and ground truths for analysis"""
        if not self.config.get('save_predictions', True):
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results = {
            'epoch': epoch,
            'metrics': metrics,
            'config': self.config,
            'results': []
        }
        
        for i, (pred, gt) in enumerate(zip(predictions[:500], ground_truths[:500])):  # Save first 500
            results['results'].append({
                'index': i,
                'prediction': pred,
                'ground_truth': gt,
                'normalized_prediction': normalize_vietnamese_answer(pred),
                'normalized_ground_truth': normalize_vietnamese_answer(gt)
            })
        
        # Save to JSON
        results_file = f'predictions_epoch_{epoch+1}_{timestamp}.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"Predictions saved to: {results_file}")
    
    def train(self, num_epochs):
        """Enhanced full training loop"""
        print(f"Starting enhanced training for {num_epochs} epochs:")
        print(f"  Stage 1 (Frozen encoders): epochs 1-{self.config['stage1_epochs']}")
        print(f"  Stage 2 (Partial unfreeze): epochs {self.config['stage1_epochs']+1}-{num_epochs}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 80)
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Evaluate
            val_metrics, predictions, ground_truths = self.evaluate_vqa()
            
            # Print comprehensive results
            print(f"\nEpoch {epoch + 1} Results:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Validation Metrics:")
            print(f"    Exact Match Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"    Fuzzy Accuracy: {val_metrics['fuzzy_accuracy']:.4f}")
            print(f"    Precision: {val_metrics['precision']:.4f}")
            print(f"    Recall: {val_metrics['recall']:.4f}")
            print(f"    F1 Score: {val_metrics['f1_score']:.4f}")
            
            if 'bleu_1' in val_metrics:
                print(f"    BLEU-1: {val_metrics['bleu_1']:.4f}")
                print(f"    BLEU-4: {val_metrics['bleu_4']:.4f}")
                print(f"    ROUGE-L: {val_metrics['rouge_l']:.4f}")
            
            # Log to wandb
            if self.use_wandb:
                log_dict = {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    **{f'val_{k}': v for k, v in val_metrics.items() if isinstance(v, (int, float))}
                }
                wandb.log(log_dict)
            
            # Check if this is the best model
            is_best = val_metrics['fuzzy_accuracy'] > self.best_fuzzy_accuracy
            if is_best:
                self.best_fuzzy_accuracy = val_metrics['fuzzy_accuracy']
                self.best_accuracy = val_metrics['accuracy']
                self.best_f1 = val_metrics['f1_score']
            
            # Save checkpoint
            if (epoch + 1) % self.config.get('save_every_n_epochs', 1) == 0:
                checkpoint_path = self.save_checkpoint(epoch, val_metrics, is_best)
                print(f"Checkpoint saved: {os.path.basename(checkpoint_path)}")
            
            # Save predictions
            self.save_predictions(predictions, ground_truths, epoch, val_metrics)
            
            if is_best:
                print(f"🎉 New best fuzzy accuracy: {self.best_fuzzy_accuracy:.4f}")
        
        # Final summary
        print(f"\n{'='*80}")
        print(f"TRAINING COMPLETED!")
        print(f"{'='*80}")
        print(f"Best Results:")
        print(f"  Fuzzy Accuracy: {self.best_fuzzy_accuracy:.4f}")
        print(f"  Exact Match Accuracy: {self.best_accuracy:.4f}")
        print(f"  F1 Score: {self.best_f1:.4f}")
        
        if self.use_wandb:
            wandb.log({
                'final_best_fuzzy_accuracy': self.best_fuzzy_accuracy,
                'final_best_accuracy': self.best_accuracy,
                'final_best_f1': self.best_f1
            })
            wandb.finish()
        
        return self.best_fuzzy_accuracy