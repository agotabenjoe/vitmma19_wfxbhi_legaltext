#!/usr/bin/env python3
"""
Label Studio Consensus Analysis Script

This script analyzes consensus labeling from Label Studio JSON exports.
It calculates:
- Percentage of labels with perfect alignment vs disagreement
- Examples of both categories
- Accuracy of individual labels compared to majority vote
"""

import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style for better looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class LabelStudioConsensusAnalyzer:
    """Analyzes consensus in Label Studio annotations."""
    
    def __init__(self, folder_path: str):
        """
        Initialize the analyzer.
        
        Args:
            folder_path: Path to folder containing Label Studio JSON exports
        """
        self.folder_path = Path(folder_path)
        self.data = []
        self.task_annotations = defaultdict(list)  # Now keyed by text content, not task ID
        
    def load_json_files(self) -> None:
        """Load all JSON files from the specified folder."""
        json_files = list(self.folder_path.glob("*.json"))
        
        if not json_files:
            print(f"No JSON files found in {self.folder_path}")
            return
        
        print(f"Found {len(json_files)} JSON file(s)")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Tag each item with the source file (annotator)
                    if isinstance(data, list):
                        for item in data:
                            item['_source_file'] = json_file.stem  # filename without extension
                        self.data.extend(data)
                    else:
                        data['_source_file'] = json_file.stem
                        self.data.append(data)
                print(f"Loaded: {json_file.name}")
            except Exception as e:
                print(f"Error loading {json_file.name}: {e}")
    
    def extract_annotations(self) -> None:
        """Extract annotations and group by task TEXT CONTENT (not ID)."""
        for item in self.data:
            task_id = item.get('id')
            text = item.get('data', {}).get('text', '')
            source_file = item.get('_source_file', 'unknown')  # Get the source file (annotator)
            
            # Normalize text as key (strip whitespace, lowercase for matching)
            text_key = text.strip()
            
            annotations = item.get('annotations', [])
            for annotation in annotations:
                if annotation.get('result'):
                    for result in annotation['result']:
                        if result.get('type') == 'choices':
                            label = result.get('value', {}).get('choices', [])
                            if label:
                                self.task_annotations[text_key].append({
                                    'text': text,
                                    'task_id': task_id,
                                    'label': label[0],
                                    'annotator_id': source_file,  # Use source file as annotator ID
                                    'annotation_id': annotation.get('id')
                                })
    
    def calculate_label_confusion_matrix(self) -> Tuple[np.ndarray, list]:
        """
        Create a confusion matrix showing which labels are confused with majority labels.
        
        Returns:
            Tuple of (confusion_matrix, label_list)
        """
        # Collect all labels
        all_labels_set = set()
        label_pairs = []  # (given_label, majority_label)
        
        for text_key, annotations in self.task_annotations.items():
            if len(annotations) < 2:
                continue
            
            # Determine majority label
            labels = [ann['label'] for ann in annotations]
            label_counts = Counter(labels)
            majority_label = label_counts.most_common(1)[0][0]
            
            # Collect all unique labels
            for ann in annotations:
                all_labels_set.add(ann['label'])
                label_pairs.append((ann['label'], majority_label))
        
        # Sort labels for consistency
        all_labels = sorted(list(all_labels_set))
        
        # Create confusion matrix
        n_labels = len(all_labels)
        confusion_matrix = np.zeros((n_labels, n_labels))
        
        for given_label, majority_label in label_pairs:
            given_idx = all_labels.index(given_label)
            majority_idx = all_labels.index(majority_label)
            confusion_matrix[majority_idx, given_idx] += 1
        
        return confusion_matrix, all_labels
    
    def calculate_majority_agreement_stats(self) -> Dict:
        """
        Calculate statistics about majority agreement strength.
        
        Returns:
            Dictionary with majority agreement statistics
        """
        agreement_stats = {
            'unanimous': 0,  # All annotators agree
            'strong_majority': 0,  # >75% agree
            'majority': 0,  # >50% agree
            'tie': 0,  # Equal split
            'total_with_multiple': 0,
            'avg_agreement_rate': [],
            'majority_confidence': [],
            'total_annotations': 0,
            'correct_annotations': 0
        }
        
        for text_key, annotations in self.task_annotations.items():
            if len(annotations) < 2:
                continue
            
            agreement_stats['total_with_multiple'] += 1
            
            labels = [ann['label'] for ann in annotations]
            label_counts = Counter(labels)
            most_common = label_counts.most_common(1)[0]
            majority_count = most_common[1]
            majority_label = most_common[0]
            total_count = len(labels)
            
            agreement_rate = majority_count / total_count
            agreement_stats['avg_agreement_rate'].append(agreement_rate)
            agreement_stats['majority_confidence'].append(majority_count)
            
            # Calculate how many annotations match the majority
            for ann in annotations:
                agreement_stats['total_annotations'] += 1
                if ann['label'] == majority_label:
                    agreement_stats['correct_annotations'] += 1
            
            # Categorize agreement strength
            if agreement_rate == 1.0:
                agreement_stats['unanimous'] += 1
            elif agreement_rate > 0.75:
                agreement_stats['strong_majority'] += 1
            elif agreement_rate > 0.5:
                agreement_stats['majority'] += 1
            else:
                agreement_stats['tie'] += 1
        
        # Calculate averages
        if agreement_stats['avg_agreement_rate']:
            agreement_stats['mean_agreement_rate'] = sum(agreement_stats['avg_agreement_rate']) / len(agreement_stats['avg_agreement_rate'])
            agreement_stats['median_agreement_rate'] = sorted(agreement_stats['avg_agreement_rate'])[len(agreement_stats['avg_agreement_rate']) // 2]
        else:
            agreement_stats['mean_agreement_rate'] = 0
            agreement_stats['median_agreement_rate'] = 0
        
        # Calculate overall annotation accuracy
        if agreement_stats['total_annotations'] > 0:
            agreement_stats['overall_annotation_accuracy'] = (agreement_stats['correct_annotations'] / agreement_stats['total_annotations']) * 100
        else:
            agreement_stats['overall_annotation_accuracy'] = 0
        
        return agreement_stats
    
    def calculate_consensus(self) -> Dict:
        """
        Calculate consensus metrics for all tasks (matched by text content).
        
        Returns:
            Dictionary with consensus statistics
        """
        perfect_alignment = []
        disagreement = []
        
        for text_key, annotations in self.task_annotations.items():
            if len(annotations) < 2:
                continue
            
            labels = [ann['label'] for ann in annotations]
            label_counts = Counter(labels)
            
            # Get unique task IDs for this text (might be multiple if same text appears with different IDs)
            task_ids = list(set([ann['task_id'] for ann in annotations]))
            
            # Perfect alignment: all annotators agree
            if len(label_counts) == 1:
                perfect_alignment.append({
                    'task_ids': task_ids,
                    'text': annotations[0]['text'],
                    'label': labels[0],
                    'num_annotators': len(annotations)
                })
            else:
                # Disagreement
                disagreement.append({
                    'task_ids': task_ids,
                    'text': annotations[0]['text'],
                    'labels': dict(label_counts),
                    'num_annotators': len(annotations)
                })
        
        total_tasks = len(perfect_alignment) + len(disagreement)
        
        return {
            'total_tasks': total_tasks,
            'perfect_alignment': perfect_alignment,
            'disagreement': disagreement,
            'perfect_alignment_pct': (len(perfect_alignment) / total_tasks * 100) if total_tasks > 0 else 0,
            'disagreement_pct': (len(disagreement) / total_tasks * 100) if total_tasks > 0 else 0
        }
    
    def calculate_annotator_accuracy(self) -> pd.DataFrame:
        """
        Calculate accuracy of each annotator compared to majority vote (matched by text content).
        
        Returns:
            DataFrame with annotator accuracy metrics
        """
        annotator_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'labels': []})
        
        for text_key, annotations in self.task_annotations.items():
            if len(annotations) < 2:
                continue
            
            # Determine majority label
            labels = [ann['label'] for ann in annotations]
            label_counts = Counter(labels)
            majority_label = label_counts.most_common(1)[0][0]
            
            # Check each annotator's label against majority
            for ann in annotations:
                annotator_id = ann['annotator_id']
                is_correct = ann['label'] == majority_label
                
                annotator_stats[annotator_id]['correct'] += int(is_correct)
                annotator_stats[annotator_id]['total'] += 1
                annotator_stats[annotator_id]['labels'].append({
                    'task_id': ann['task_id'],
                    'text': ann['text'],
                    'given_label': ann['label'],
                    'majority_label': majority_label,
                    'is_correct': is_correct
                })
        
        # Convert to DataFrame
        results = []
        for annotator_id, stats in annotator_stats.items():
            accuracy = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
            results.append({
                'annotator_id': annotator_id,
                'correct': stats['correct'],
                'total': stats['total'],
                'accuracy_pct': accuracy
            })
        
        df = pd.DataFrame(results).sort_values('accuracy_pct', ascending=False)
        return df, annotator_stats
    
    def print_report(self) -> None:
        """Print a comprehensive analysis report."""
        print("\n" + "="*80)
        print("LABEL STUDIO CONSENSUS ANALYSIS REPORT")
        print("="*80)
        
        # Consensus metrics
        consensus_results = self.calculate_consensus()
        
        print(f"\n📊 CONSENSUS METRICS")
        print(f"{'─'*80}")
        print(f"Total tasks with multiple annotations: {consensus_results['total_tasks']}")
        print(f"Tasks with perfect alignment: {len(consensus_results['perfect_alignment'])} ({consensus_results['perfect_alignment_pct']:.2f}%)")
        print(f"Tasks with disagreement: {len(consensus_results['disagreement'])} ({consensus_results['disagreement_pct']:.2f}%)")
        
        # Majority agreement statistics
        majority_stats = self.calculate_majority_agreement_stats()
        
        print(f"\n🎯 MAJORITY AGREEMENT STRENGTH")
        print(f"{'─'*80}")
        if majority_stats['total_with_multiple'] > 0:
            print(f"Unanimous (100% agree): {majority_stats['unanimous']} ({majority_stats['unanimous']/majority_stats['total_with_multiple']*100:.1f}%)")
            print(f"Strong Majority (>75% agree): {majority_stats['strong_majority']} ({majority_stats['strong_majority']/majority_stats['total_with_multiple']*100:.1f}%)")
            print(f"Majority (>50% agree): {majority_stats['majority']} ({majority_stats['majority']/majority_stats['total_with_multiple']*100:.1f}%)")
            print(f"Tie/No Clear Majority: {majority_stats['tie']} ({majority_stats['tie']/majority_stats['total_with_multiple']*100:.1f}%)")
            print(f"\n📈 Average Agreement Rate: {majority_stats['mean_agreement_rate']*100:.2f}%")
            print(f"📊 Median Agreement Rate: {majority_stats['median_agreement_rate']*100:.2f}%")
            print(f"\n⭐ OVERALL ANNOTATION ACCURACY (vs Majority): {majority_stats['overall_annotation_accuracy']:.2f}%")
            print(f"   Total Annotations: {majority_stats['total_annotations']}")
            print(f"   Correct (Match Majority): {majority_stats['correct_annotations']}")
            print(f"   Incorrect (Disagree with Majority): {majority_stats['total_annotations'] - majority_stats['correct_annotations']}")
        else:
            print("No tasks with multiple annotations found.")
        
        # Label confusion analysis
        print(f"\n🔀 LABEL CONFUSION ANALYSIS")
        print(f"{'─'*80}")
        confusion_matrix, label_list = self.calculate_label_confusion_matrix()
        
        if confusion_matrix is not None and len(label_list) > 1:
            print(f"Total unique labels: {len(label_list)}")
            print(f"Labels: {', '.join(label_list)}")
            
            # Find most confused label pairs (off-diagonal)
            confused_pairs = []
            for i in range(len(label_list)):
                for j in range(len(label_list)):
                    if i != j and confusion_matrix[i, j] > 0:
                        confused_pairs.append({
                            'majority': label_list[i],
                            'given': label_list[j],
                            'count': int(confusion_matrix[i, j])
                        })
            
            confused_pairs.sort(key=lambda x: x['count'], reverse=True)
            
            if confused_pairs:
                print(f"\n🔴 Top Label Confusions (most common mistakes):")
                for idx, pair in enumerate(confused_pairs[:5], 1):
                    print(f"  {idx}. When majority is '{pair['majority']}', annotators gave '{pair['given']}' {pair['count']} times")
            else:
                print("\n✅ No label confusions found - all annotations match majority!")
        else:
            print("Not enough data for confusion analysis.")
        
        # Examples of perfect alignment
        print(f"\n✅ EXAMPLES OF PERFECT ALIGNMENT")
        print(f"{'─'*80}")
        for i, example in enumerate(consensus_results['perfect_alignment'][:3], 1):
            print(f"\nExample {i}:")
            print(f"  Task IDs: {example['task_ids']}")
            print(f"  Text: {example['text'][:100]}..." if len(example['text']) > 100 else f"  Text: {example['text']}")
            print(f"  Agreed Label: {example['label']}")
            print(f"  Number of Annotators: {example['num_annotators']}")
        
        # Examples of disagreement
        print(f"\n❌ EXAMPLES OF DISAGREEMENT")
        print(f"{'─'*80}")
        for i, example in enumerate(consensus_results['disagreement'][:3], 1):
            print(f"\nExample {i}:")
            print(f"  Task IDs: {example['task_ids']}")
            print(f"  Text: {example['text'][:100]}..." if len(example['text']) > 100 else f"  Text: {example['text']}")
            print(f"  Label Distribution: {example['labels']}")
            print(f"  Number of Annotators: {example['num_annotators']}")
        
        # Annotator accuracy
        print(f"\n👥 ANNOTATOR ACCURACY (vs Majority Vote)")
        print(f"{'─'*80}")
        accuracy_df, annotator_details = self.calculate_annotator_accuracy()
        
        if not accuracy_df.empty:
            print(accuracy_df.to_string(index=False))
            
            # Show examples of disagreements for each annotator
            print(f"\n📋 DETAILED DISAGREEMENT EXAMPLES BY ANNOTATOR")
            print(f"{'─'*80}")
            for annotator_id, stats in annotator_details.items():
                disagreements = [l for l in stats['labels'] if not l['is_correct']]
                if disagreements:
                    print(f"\nAnnotator {annotator_id} - {len(disagreements)} disagreement(s):")
                    for i, dis in enumerate(disagreements[:3], 1):
                        task_text = dis['text']
                        print(f"  {i}. Task {dis['task_id']}")
                        print(f"     Text: {task_text[:80]}..." if len(task_text) > 80 else f"     Text: {task_text}")
                        print(f"     Given: {dis['given_label']}")
                        print(f"     Majority: {dis['majority_label']}")
        else:
            print("No annotator accuracy data available (need multiple annotations per task)")
        
        # Show ALL annotators and their task counts
        print(f"\n📊 ALL ANNOTATORS SUMMARY")
        print(f"{'─'*80}")
        all_annotators = defaultdict(lambda: {'total_tasks': 0, 'shared_tasks': 0, 'unique_tasks': 0})
        
        for text_key, annotations in self.task_annotations.items():
            annotator_ids = [ann['annotator_id'] for ann in annotations]
            for annotator_id in set(annotator_ids):
                all_annotators[annotator_id]['total_tasks'] += 1
                if len(annotations) > 1:
                    all_annotators[annotator_id]['shared_tasks'] += 1
                else:
                    all_annotators[annotator_id]['unique_tasks'] += 1
        
        all_annotators_df = pd.DataFrame([
            {
                'annotator_id': ann_id,
                'total_tasks': stats['total_tasks'],
                'shared_tasks': stats['shared_tasks'],
                'unique_tasks': stats['unique_tasks'],
                'accuracy_pct': accuracy_df[accuracy_df['annotator_id'] == ann_id]['accuracy_pct'].values[0] 
                               if ann_id in accuracy_df['annotator_id'].values else 'N/A (no shared tasks)'
            }
            for ann_id, stats in all_annotators.items()
        ]).sort_values('total_tasks', ascending=False)
        
        print(all_annotators_df.to_string(index=False))
        
        print(f"\n{'='*80}")
        print("END OF REPORT")
        print(f"{'='*80}\n")
    
    def generate_visualizations(self, output_folder: str = ".") -> None:
        """
        Generate beautiful visualizations of the consensus analysis.
        
        Args:
            output_folder: Folder to save visualization files
        """
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True)
        
        print("\n🎨 GENERATING VISUALIZATIONS...")
        print(f"{'─'*80}")
        
        # Get data
        consensus_results = self.calculate_consensus()
        accuracy_df, annotator_details = self.calculate_annotator_accuracy()
        
        # 1. CONSENSUS PIE CHART
        if consensus_results['total_tasks'] > 0:
            fig, ax = plt.subplots(figsize=(10, 8))
            sizes = [len(consensus_results['perfect_alignment']), 
                    len(consensus_results['disagreement'])]
            labels = [f"Perfect Alignment\n({consensus_results['perfect_alignment_pct']:.1f}%)", 
                     f"Disagreement\n({consensus_results['disagreement_pct']:.1f}%)"]
            colors = ['#2ecc71', '#e74c3c']
            explode = (0.05, 0.05)
            
            ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.0f%%',
                   shadow=True, startangle=90, textprops={'fontsize': 14, 'weight': 'bold'})
            ax.set_title('Consensus Analysis: Agreement vs Disagreement', 
                        fontsize=16, weight='bold', pad=20)
            
            plt.tight_layout()
            plt.savefig(output_path / 'consensus_pie_chart.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Saved: consensus_pie_chart.png")
        
        # 1B. MAJORITY AGREEMENT STRENGTH BAR CHART
        majority_stats = self.calculate_majority_agreement_stats()
        if majority_stats['total_with_multiple'] > 0:
            fig, ax = plt.subplots(figsize=(12, 7))
            
            categories = ['Unanimous\n(100%)', 'Strong Majority\n(>75%)', 
                         'Majority\n(>50%)', 'Tie/Split\n(≤50%)']
            counts = [
                majority_stats['unanimous'],
                majority_stats['strong_majority'],
                majority_stats['majority'],
                majority_stats['tie']
            ]
            percentages = [c / majority_stats['total_with_multiple'] * 100 for c in counts]
            
            colors = ['#27ae60', '#2ecc71', '#f39c12', '#e74c3c']
            bars = ax.bar(categories, counts, color=colors, edgecolor='black', linewidth=1.5)
            
            # Add value labels
            for bar, count, pct in zip(bars, counts, percentages):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{count}\n({pct:.1f}%)',
                       ha='center', va='bottom', fontsize=11, weight='bold')
            
            ax.set_ylabel('Number of Tasks', fontsize=13, weight='bold')
            ax.set_title(f'Majority Agreement Strength Distribution\n(Avg Agreement: {majority_stats["mean_agreement_rate"]*100:.1f}%)', 
                        fontsize=15, weight='bold', pad=20)
            ax.set_ylim(0, max(counts) * 1.15)
            
            # Add grid
            ax.yaxis.grid(True, alpha=0.3)
            ax.set_axisbelow(True)
            
            plt.tight_layout()
            plt.savefig(output_path / 'majority_agreement_strength.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Saved: majority_agreement_strength.png")
            
            # 1C. Agreement Rate Distribution (Histogram)
            fig, ax = plt.subplots(figsize=(12, 6))
            
            agreement_rates = [rate * 100 for rate in majority_stats['avg_agreement_rate']]
            
            ax.hist(agreement_rates, bins=20, color='#3498db', edgecolor='black', linewidth=1.2, alpha=0.7)
            ax.axvline(majority_stats['mean_agreement_rate'] * 100, color='red', 
                      linestyle='--', linewidth=2, label=f'Mean: {majority_stats["mean_agreement_rate"]*100:.1f}%')
            ax.axvline(majority_stats['median_agreement_rate'] * 100, color='green', 
                      linestyle='--', linewidth=2, label=f'Median: {majority_stats["median_agreement_rate"]*100:.1f}%')
            
            ax.set_xlabel('Agreement Rate (%)', fontsize=13, weight='bold')
            ax.set_ylabel('Number of Tasks', fontsize=13, weight='bold')
            ax.set_title('Distribution of Agreement Rates Across Tasks', fontsize=15, weight='bold', pad=20)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path / 'agreement_rate_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Saved: agreement_rate_distribution.png")
        
        # 1D. LABEL CONFUSION MATRIX (Overall - all annotations vs majority)
        confusion_matrix, label_list = self.calculate_label_confusion_matrix()
        
        if confusion_matrix is not None and len(label_list) > 1:
            fig, ax = plt.subplots(figsize=(max(10, len(label_list) * 1.5), max(8, len(label_list) * 1.2)))
            
            # Create heatmap
            sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='YlOrRd', 
                       xticklabels=label_list, yticklabels=label_list,
                       cbar_kws={'label': 'Count'}, ax=ax, square=True,
                       linewidths=1, linecolor='black')
            
            ax.set_title('Label Confusion Matrix: Given Labels vs Majority Labels\n(Rows=Majority, Columns=Given)', 
                        fontsize=15, weight='bold', pad=20)
            ax.set_xlabel('Given Labels (What annotators chose)', fontsize=13, weight='bold')
            ax.set_ylabel('Majority Labels (Consensus)', fontsize=13, weight='bold')
            
            # Rotate labels for better readability
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            # Add diagonal highlight to show correct predictions
            for i in range(len(label_list)):
                ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='green', lw=3))
            
            plt.tight_layout()
            plt.savefig(output_path / 'label_confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Saved: label_confusion_matrix.png")
            
            # Also create a normalized version (percentages)
            fig, ax = plt.subplots(figsize=(max(10, len(label_list) * 1.5), max(8, len(label_list) * 1.2)))
            
            # Normalize by row (by majority label)
            confusion_normalized = np.zeros_like(confusion_matrix, dtype=float)
            for i in range(len(label_list)):
                row_sum = confusion_matrix[i, :].sum()
                if row_sum > 0:
                    confusion_normalized[i, :] = confusion_matrix[i, :] / row_sum * 100
            
            sns.heatmap(confusion_normalized, annot=True, fmt='.1f', cmap='YlOrRd',
                       xticklabels=label_list, yticklabels=label_list,
                       cbar_kws={'label': 'Percentage (%)'}, ax=ax, square=True,
                       linewidths=1, linecolor='black', vmin=0, vmax=100)
            
            ax.set_title('Label Confusion Matrix (Normalized %)\nShows: For each majority label, what % were annotated as each label', 
                        fontsize=15, weight='bold', pad=20)
            ax.set_xlabel('Given Labels (What annotators chose)', fontsize=13, weight='bold')
            ax.set_ylabel('Majority Labels (Consensus)', fontsize=13, weight='bold')
            
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            # Add diagonal highlight
            for i in range(len(label_list)):
                ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='green', lw=3))
            
            plt.tight_layout()
            plt.savefig(output_path / 'label_confusion_matrix_normalized.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Saved: label_confusion_matrix_normalized.png")
        
        # 2. ANNOTATOR ACCURACY BAR CHART
        if not accuracy_df.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            bars = ax.bar(accuracy_df['annotator_id'].astype(str), 
                         accuracy_df['accuracy_pct'],
                         color=['#3498db' if x >= 70 else '#e67e22' if x >= 50 else '#e74c3c' 
                               for x in accuracy_df['accuracy_pct']])
            
            ax.set_xlabel('Annotator ID', fontsize=12, weight='bold')
            ax.set_ylabel('Accuracy (%)', fontsize=12, weight='bold')
            ax.set_title('Annotator Accuracy vs Majority Vote', fontsize=14, weight='bold')
            ax.set_ylim(0, 105)
            ax.axhline(y=70, color='g', linestyle='--', alpha=0.5, label='70% threshold')
            ax.legend()
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=10, weight='bold')
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(output_path / 'annotator_accuracy_bar.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Saved: annotator_accuracy_bar.png")
        
        # 3. ANNOTATOR TASK DISTRIBUTION
        all_annotators = defaultdict(lambda: {'total_tasks': 0, 'shared_tasks': 0, 'unique_tasks': 0})
        
        for text_key, annotations in self.task_annotations.items():
            annotator_ids = [ann['annotator_id'] for ann in annotations]
            for annotator_id in set(annotator_ids):
                all_annotators[annotator_id]['total_tasks'] += 1
                if len(annotations) > 1:
                    all_annotators[annotator_id]['shared_tasks'] += 1
                else:
                    all_annotators[annotator_id]['unique_tasks'] += 1
        
        if all_annotators:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            annotator_ids = list(all_annotators.keys())
            shared_tasks = [all_annotators[aid]['shared_tasks'] for aid in annotator_ids]
            unique_tasks = [all_annotators[aid]['unique_tasks'] for aid in annotator_ids]
            
            x = range(len(annotator_ids))
            width = 0.35
            
            bars1 = ax.bar([i - width/2 for i in x], shared_tasks, width, 
                          label='Shared Tasks', color='#3498db')
            bars2 = ax.bar([i + width/2 for i in x], unique_tasks, width,
                          label='Unique Tasks', color='#9b59b6')
            
            ax.set_xlabel('Annotator ID', fontsize=12, weight='bold')
            ax.set_ylabel('Number of Tasks', fontsize=12, weight='bold')
            ax.set_title('Task Distribution per Annotator', fontsize=14, weight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([str(aid) for aid in annotator_ids], rotation=45, ha='right')
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(output_path / 'task_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Saved: task_distribution.png")
        
        # 4. LABEL DISTRIBUTION HEATMAP (if we have disagreements)
        if consensus_results['disagreement']:
            # Collect all labels from disagreements
            label_combinations = []
            for dis in consensus_results['disagreement']:
                label_combinations.append(dis['labels'])
            
            # Get unique labels
            all_labels = set()
            for labels_dict in label_combinations:
                all_labels.update(labels_dict.keys())
            all_labels = sorted(list(all_labels))
            
            if len(all_labels) > 1:
                # Create confusion-style matrix
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Count co-occurrences
                label_counts = Counter()
                for labels_dict in label_combinations:
                    for label in labels_dict:
                        label_counts[label] += labels_dict[label]
                
                # Create bar chart of label frequency
                labels_sorted = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
                labels_list = [l[0] for l in labels_sorted]
                counts_list = [l[1] for l in labels_sorted]
                
                bars = ax.barh(labels_list, counts_list, color='#e74c3c')
                ax.set_xlabel('Frequency', fontsize=12, weight='bold')
                ax.set_ylabel('Label', fontsize=12, weight='bold')
                ax.set_title('Label Distribution in Disagreements', fontsize=14, weight='bold')
                
                # Add value labels
                for i, (bar, count) in enumerate(zip(bars, counts_list)):
                    ax.text(count, bar.get_y() + bar.get_height()/2.,
                           f' {count}',
                           ha='left', va='center', fontsize=10, weight='bold')
                
                plt.tight_layout()
                plt.savefig(output_path / 'label_distribution.png', dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✅ Saved: label_distribution.png")
        
        # 5. COMPREHENSIVE DASHBOARD
        if consensus_results['total_tasks'] > 0 and not accuracy_df.empty:
            fig = plt.figure(figsize=(16, 10))
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
            
            # Subplot 1: Consensus pie
            ax1 = fig.add_subplot(gs[0, 0])
            sizes = [len(consensus_results['perfect_alignment']), 
                    len(consensus_results['disagreement'])]
            colors = ['#2ecc71', '#e74c3c']
            ax1.pie(sizes, labels=['Perfect\nAlignment', 'Disagreement'], 
                   colors=colors, autopct='%1.1f%%', startangle=90)
            ax1.set_title('Consensus Overview', fontsize=12, weight='bold')
            
            # Subplot 2: Accuracy bar
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.bar(accuracy_df['annotator_id'].astype(str), 
                   accuracy_df['accuracy_pct'],
                   color='#3498db')
            ax2.set_xlabel('Annotator', fontsize=10)
            ax2.set_ylabel('Accuracy (%)', fontsize=10)
            ax2.set_title('Annotator Accuracy', fontsize=12, weight='bold')
            ax2.tick_params(axis='x', rotation=45)
            
            # Subplot 3: Task counts
            ax3 = fig.add_subplot(gs[1, 0])
            annotator_ids = list(all_annotators.keys())
            total_tasks = [all_annotators[aid]['total_tasks'] for aid in annotator_ids]
            ax3.bar([str(aid) for aid in annotator_ids], total_tasks, color='#9b59b6')
            ax3.set_xlabel('Annotator', fontsize=10)
            ax3.set_ylabel('Total Tasks', fontsize=10)
            ax3.set_title('Tasks per Annotator', fontsize=12, weight='bold')
            ax3.tick_params(axis='x', rotation=45)
            
            # Subplot 4: Summary stats
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.axis('off')
            
            majority_stats = self.calculate_majority_agreement_stats()
            
            summary_text = f"""
            📊 SUMMARY STATISTICS
            
            Total Tasks Analyzed: {consensus_results['total_tasks']}
            Perfect Alignment: {len(consensus_results['perfect_alignment'])}
            Disagreements: {len(consensus_results['disagreement'])}
            
            Average Agreement Rate: {majority_stats['mean_agreement_rate']*100:.1f}%
            Median Agreement Rate: {majority_stats['median_agreement_rate']*100:.1f}%
            
            ⭐ Overall Annotation Accuracy: {majority_stats['overall_annotation_accuracy']:.1f}%
            
            Total Annotators: {len(all_annotators)}
            Avg Annotator Accuracy: {accuracy_df['accuracy_pct'].mean():.1f}%
            Best Annotator Accuracy: {accuracy_df['accuracy_pct'].max():.1f}%
            """
            
            ax4.text(0.1, 0.5, summary_text, fontsize=11, 
                    verticalalignment='center', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            
            fig.suptitle('Label Studio Consensus Analysis Dashboard', 
                        fontsize=16, weight='bold')
            
            plt.savefig(output_path / 'dashboard.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Saved: dashboard.png")
        
        print(f"\n🎉 ALL VISUALIZATIONS GENERATED!")
        print(f"{'─'*80}\n")
    
    def generate_visualizations(self, output_folder: str = ".") -> None:
        """
        Save analysis results to CSV files.
        
        Args:
            output_folder: Folder to save CSV files
        """
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True)
        
        # Save consensus results
        consensus_results = self.calculate_consensus()
        
        # Perfect alignment
        if consensus_results['perfect_alignment']:
            df_perfect = pd.DataFrame(consensus_results['perfect_alignment'])
            df_perfect.to_csv(output_path / 'perfect_alignment.csv', index=False, encoding='utf-8')
            print(f"Saved: {output_path / 'perfect_alignment.csv'}")
        
        # Disagreement
        if consensus_results['disagreement']:
            df_disagreement = pd.DataFrame(consensus_results['disagreement'])
            df_disagreement.to_csv(output_path / 'disagreement.csv', index=False, encoding='utf-8')
            print(f"Saved: {output_path / 'disagreement.csv'}")
        
        # Annotator accuracy
        accuracy_df, _ = self.calculate_annotator_accuracy()
        if not accuracy_df.empty:
            accuracy_df.to_csv(output_path / 'annotator_accuracy.csv', index=False, encoding='utf-8')
            print(f"Saved: {output_path / 'annotator_accuracy.csv'}")
    
    def generate_visualizations(self, output_folder: str = ".") -> None:
        """
        Generate comprehensive visualizations of the analysis.
        
        Args:
            output_folder: Folder to save visualization files
        """
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        
        consensus_results = self.calculate_consensus()
        accuracy_df, annotator_details = self.calculate_annotator_accuracy()
        
        # 1. Consensus Pie Chart
        fig, ax = plt.subplots(figsize=(10, 8))
        sizes = [len(consensus_results['perfect_alignment']), len(consensus_results['disagreement'])]
        labels = ['Perfect Alignment', 'Disagreement']
        colors = ['#2ecc71', '#e74c3c']
        explode = (0.05, 0.05)
        
        if sum(sizes) > 0:
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                                startangle=90, explode=explode, textprops={'fontsize': 12})
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_weight('bold')
                autotext.set_fontsize(14)
            
            ax.set_title('Consensus Distribution Across All Tasks', fontsize=16, weight='bold', pad=20)
            plt.tight_layout()
            plt.savefig(output_path / 'consensus_distribution.png', dpi=300, bbox_inches='tight')
            print(f"Saved: {output_path / 'consensus_distribution.png'}")
            plt.close()
        
        # 2. Annotator Accuracy Bar Chart
        if not accuracy_df.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            colors_bar = ['#3498db' if acc >= 80 else '#e67e22' if acc >= 60 else '#e74c3c' 
                          for acc in accuracy_df['accuracy_pct']]
            
            bars = ax.bar(accuracy_df['annotator_id'].astype(str), accuracy_df['accuracy_pct'], 
                         color=colors_bar, edgecolor='black', linewidth=1.2)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=10, weight='bold')
            
            ax.set_xlabel('Annotator ID', fontsize=12, weight='bold')
            ax.set_ylabel('Accuracy (%)', fontsize=12, weight='bold')
            ax.set_title('Annotator Accuracy vs Majority Vote', fontsize=16, weight='bold', pad=20)
            ax.set_ylim(0, 105)
            ax.axhline(y=80, color='green', linestyle='--', alpha=0.5, label='80% threshold')
            ax.legend()
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(output_path / 'annotator_accuracy.png', dpi=300, bbox_inches='tight')
            print(f"Saved: {output_path / 'annotator_accuracy.png'}")
            plt.close()
        
        # 3. Label Distribution Heatmap (for disagreements)
        if consensus_results['disagreement']:
            # Collect all labels
            all_labels_set = set()
            for dis in consensus_results['disagreement']:
                all_labels_set.update(dis['labels'].keys())
            
            all_labels = sorted(list(all_labels_set))
            
            if len(all_labels) > 1 and len(consensus_results['disagreement']) > 0:
                # Create matrix for top disagreements
                top_disagreements = consensus_results['disagreement'][:min(20, len(consensus_results['disagreement']))]
                matrix = []
                row_labels = []
                
                for i, dis in enumerate(top_disagreements):
                    row = [dis['labels'].get(label, 0) for label in all_labels]
                    matrix.append(row)
                    text_preview = dis['text'][:30] + '...' if len(dis['text']) > 30 else dis['text']
                    row_labels.append(f"Task {i+1}")
                
                if matrix:
                    fig, ax = plt.subplots(figsize=(max(10, len(all_labels) * 1.5), max(8, len(row_labels) * 0.4)))
                    sns.heatmap(matrix, annot=True, fmt='g', cmap='YlOrRd', 
                               xticklabels=all_labels, yticklabels=row_labels, 
                               cbar_kws={'label': 'Number of Annotations'}, ax=ax)
                    ax.set_title('Label Distribution for Disagreement Tasks', fontsize=16, weight='bold', pad=20)
                    ax.set_xlabel('Labels', fontsize=12, weight='bold')
                    ax.set_ylabel('Tasks', fontsize=12, weight='bold')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    plt.savefig(output_path / 'disagreement_heatmap.png', dpi=300, bbox_inches='tight')
                    print(f"Saved: {output_path / 'disagreement_heatmap.png'}")
                    plt.close()
        
        # 4. Overall Label Distribution
        all_task_labels = []
        for text_key, annotations in self.task_annotations.items():
            for ann in annotations:
                all_task_labels.append(ann['label'])
        
        if all_task_labels:
            label_counts = Counter(all_task_labels)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            labels_list = list(label_counts.keys())
            counts_list = list(label_counts.values())
            
            colors_dist = sns.color_palette("husl", len(labels_list))
            bars = ax.bar(range(len(labels_list)), counts_list, color=colors_dist, edgecolor='black', linewidth=1.2)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', fontsize=10, weight='bold')
            
            ax.set_xlabel('Labels', fontsize=12, weight='bold')
            ax.set_ylabel('Count', fontsize=12, weight='bold')
            ax.set_title('Overall Label Distribution Across All Annotations', fontsize=16, weight='bold', pad=20)
            ax.set_xticks(range(len(labels_list)))
            ax.set_xticklabels(labels_list, rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(output_path / 'label_distribution.png', dpi=300, bbox_inches='tight')
            print(f"Saved: {output_path / 'label_distribution.png'}")
            plt.close()
        
        # 5. Annotator Task Count Comparison
        all_annotators = defaultdict(lambda: {'total_tasks': 0, 'shared_tasks': 0, 'unique_tasks': 0})
        
        for text_key, annotations in self.task_annotations.items():
            annotator_ids = [ann['annotator_id'] for ann in annotations]
            for annotator_id in set(annotator_ids):
                all_annotators[annotator_id]['total_tasks'] += 1
                if len(annotations) > 1:
                    all_annotators[annotator_id]['shared_tasks'] += 1
                else:
                    all_annotators[annotator_id]['unique_tasks'] += 1
        
        if all_annotators:
            annotators = list(all_annotators.keys())
            shared_tasks = [all_annotators[a]['shared_tasks'] for a in annotators]
            unique_tasks = [all_annotators[a]['unique_tasks'] for a in annotators]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(annotators))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, shared_tasks, width, label='Shared Tasks', 
                          color='#3498db', edgecolor='black', linewidth=1.2)
            bars2 = ax.bar(x + width/2, unique_tasks, width, label='Unique Tasks', 
                          color='#95a5a6', edgecolor='black', linewidth=1.2)
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{int(height)}', ha='center', va='bottom', fontsize=9, weight='bold')
            
            ax.set_xlabel('Annotator ID', fontsize=12, weight='bold')
            ax.set_ylabel('Number of Tasks', fontsize=12, weight='bold')
            ax.set_title('Shared vs Unique Tasks per Annotator', fontsize=16, weight='bold', pad=20)
            ax.set_xticks(x)
            ax.set_xticklabels(annotators, rotation=45, ha='right')
            ax.legend(fontsize=11)
            plt.tight_layout()
            plt.savefig(output_path / 'annotator_task_comparison.png', dpi=300, bbox_inches='tight')
            print(f"Saved: {output_path / 'annotator_task_comparison.png'}")
            plt.close()
        
        # 6. Confusion Matrix for each annotator (if we have disagreements)
        if not accuracy_df.empty and len(annotator_details) > 0:
            for annotator_id, stats in annotator_details.items():
                if stats['labels']:
                    # Get all unique labels
                    all_labels_for_confusion = set()
                    for label_info in stats['labels']:
                        all_labels_for_confusion.add(label_info['given_label'])
                        all_labels_for_confusion.add(label_info['majority_label'])
                    
                    all_labels_sorted = sorted(list(all_labels_for_confusion))
                    
                    if len(all_labels_sorted) > 1:
                        # Create confusion matrix
                        confusion = np.zeros((len(all_labels_sorted), len(all_labels_sorted)))
                        
                        for label_info in stats['labels']:
                            given_idx = all_labels_sorted.index(label_info['given_label'])
                            majority_idx = all_labels_sorted.index(label_info['majority_label'])
                            confusion[majority_idx, given_idx] += 1
                        
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(confusion, annot=True, fmt='g', cmap='Blues', 
                                   xticklabels=all_labels_sorted, yticklabels=all_labels_sorted, 
                                   cbar_kws={'label': 'Count'}, ax=ax, square=True)
                        ax.set_title(f'Confusion Matrix: {annotator_id} vs Majority Vote', 
                                    fontsize=14, weight='bold', pad=20)
                        ax.set_xlabel('Annotator Labels', fontsize=12, weight='bold')
                        ax.set_ylabel('Majority Labels', fontsize=12, weight='bold')
                        plt.xticks(rotation=45, ha='right')
                        plt.yticks(rotation=0)
                        plt.tight_layout()
                        
                        safe_filename = str(annotator_id).replace('/', '_').replace('\\', '_')
                        plt.savefig(output_path / f'confusion_matrix_{safe_filename}.png', 
                                   dpi=300, bbox_inches='tight')
                        print(f"Saved: {output_path / f'confusion_matrix_{safe_filename}.png'}")
                        plt.close()
        
        print(f"\n✅ All visualizations saved to {output_path}")
    
    def run_analysis(self, save_csv: bool = True, output_folder: str = ".", generate_plots: bool = True) -> None:
        """
        Run complete analysis pipeline.
        
        Args:
            save_csv: Whether to save results to CSV files
            output_folder: Folder to save CSV files and visualizations
            generate_plots: Whether to generate visualization plots
        """
        self.load_json_files()
        
        if not self.data:
            print("No data loaded. Exiting.")
            return
        
        self.extract_annotations()
        
        if not self.task_annotations:
            print("No annotations found. Exiting.")
            return
        
        self.print_report()
        
        if save_csv:
            self.save_results_to_csv(output_folder)
        
        if generate_plots:
            print("\n📊 Generating visualizations...")
            self.generate_visualizations(output_folder)


def main():
    """Main function to run the analysis."""
    # Configuration
    FOLDER_PATH = "./consensus/consensus"  # Change this to your folder path
    OUTPUT_FOLDER = "./analysis_results"    # Change this to your desired output folder
    
    # Create analyzer instance
    analyzer = LabelStudioConsensusAnalyzer(FOLDER_PATH)
    
    # Run analysis with visualizations
    analyzer.run_analysis(save_csv=True, output_folder=OUTPUT_FOLDER, generate_plots=True)


if __name__ == "__main__":
    main()
