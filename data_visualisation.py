"""
Simple Keystroke Data Visualization
Analyzes raw typing data without model evaluation
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams['figure.dpi'] = 150

class SimpleKeystrokeVisualizer:
    """Simple visualizer for raw keystroke data"""
    
    def __init__(self, genuine_folder="data/genuine", impostor_folder="data/impostors"):
        self.genuine_folder = Path(genuine_folder)
        self.impostor_folder = Path(impostor_folder)
        self.all_data = []
        self.df = None
        
    def load_data(self):
        """Load all JSON files and extract basic metrics"""
        print("Loading data...")
        
        # Load genuine users
        for json_file in self.genuine_folder.glob("*.json"):
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            user_id = data['metadata']['userId']
            
            for session in data['sessions']:
                for keystroke in session['data']:
                    self.all_data.append({
                        'user': user_id,
                        'user_type': 'genuine',
                        'key': keystroke['key'],
                        'dwell_time': keystroke['dwellTime'],
                        'flight_time': keystroke['flightTime'],
                        'phrase_number': keystroke['phraseNumber']
                    })
        
        # Load impostors
        if self.impostor_folder.exists():
            for json_file in self.impostor_folder.glob("*.json"):
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                user_id = data['metadata']['userId']
                
                for session in data['sessions']:
                    for keystroke in session['data']:
                        self.all_data.append({
                            'user': user_id,
                            'user_type': 'impostor',
                            'key': keystroke['key'],
                            'dwell_time': keystroke['dwellTime'],
                            'flight_time': keystroke['flightTime'],
                            'phrase_number': keystroke['phraseNumber']
                        })
        
        self.df = pd.DataFrame(self.all_data)
        
        genuine_count = len(self.df[self.df['user_type'] == 'genuine'])
        impostor_count = len(self.df[self.df['user_type'] == 'impostor'])
        genuine_users = len(self.df[self.df['user_type'] == 'genuine']['user'].unique())
        impostor_users = len(self.df[self.df['user_type'] == 'impostor']['user'].unique())
        
        print(f"✓ Loaded {genuine_count} keystrokes from {genuine_users} genuine users")
        print(f"✓ Loaded {impostor_count} keystrokes from {impostor_users} impostors")
        return self.df
    
    def calculate_typing_speed(self):
        """Calculate typing speed per phrase"""
        speeds = []
        
        # Process genuine users
        for json_file in self.genuine_folder.glob("*.json"):
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            user_id = data['metadata']['userId']
            
            for session in data['sessions']:
                phrases = {}
                for ks in session['data']:
                    phrase_num = ks['phraseNumber']
                    if phrase_num not in phrases:
                        phrases[phrase_num] = []
                    phrases[phrase_num].append(ks)
                
                for phrase_num, keystrokes in phrases.items():
                    if len(keystrokes) > 1:
                        total_time = keystrokes[-1]['releaseTime'] - keystrokes[0]['pressTime']
                        chars_per_sec = (len(keystrokes) / total_time) * 1000
                        speeds.append({
                            'user': user_id,
                            'user_type': 'genuine',
                            'phrase_number': phrase_num,
                            'typing_speed': chars_per_sec,
                            'num_keys': len(keystrokes)
                        })
        
        # Process impostors
        if self.impostor_folder.exists():
            for json_file in self.impostor_folder.glob("*.json"):
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                user_id = data['metadata']['userId']
                
                for session in data['sessions']:
                    phrases = {}
                    for ks in session['data']:
                        phrase_num = ks['phraseNumber']
                        if phrase_num not in phrases:
                            phrases[phrase_num] = []
                        phrases[phrase_num].append(ks)
                    
                    for phrase_num, keystrokes in phrases.items():
                        if len(keystrokes) > 1:
                            total_time = keystrokes[-1]['releaseTime'] - keystrokes[0]['pressTime']
                            chars_per_sec = (len(keystrokes) / total_time) * 1000
                            speeds.append({
                                'user': user_id,
                                'user_type': 'impostor',
                                'phrase_number': phrase_num,
                                'typing_speed': chars_per_sec,
                                'num_keys': len(keystrokes)
                            })
        
        return pd.DataFrame(speeds)
    
    def print_summary_stats(self):
        """Print basic statistics"""
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        
        # Genuine users
        print("\nGENUINE USERS:")
        print("-" * 60)
        genuine_df = self.df[self.df['user_type'] == 'genuine']
        for user in genuine_df['user'].unique():
            user_data = genuine_df[genuine_df['user'] == user]
            print(f"\n{user.upper()}:")
            print(f"  Total keystrokes: {len(user_data)}")
            print(f"  Avg dwell time:   {user_data['dwell_time'].mean():.1f} ms")
            print(f"  Avg flight time:  {user_data['flight_time'].mean():.1f} ms")
            print(f"  Std dwell time:   {user_data['dwell_time'].std():.1f} ms")
            print(f"  Std flight time:  {user_data['flight_time'].std():.1f} ms")
        
        # Impostor aggregate stats
        impostor_df = self.df[self.df['user_type'] == 'impostor']
        if len(impostor_df) > 0:
            print("\n\nIMPOSTORS (AGGREGATE):")
            print("-" * 60)
            print(f"  Total impostors:  {len(impostor_df['user'].unique())}")
            print(f"  Total keystrokes: {len(impostor_df)}")
            print(f"  Avg dwell time:   {impostor_df['dwell_time'].mean():.1f} ms")
            print(f"  Avg flight time:  {impostor_df['flight_time'].mean():.1f} ms")
            print(f"  Std dwell time:   {impostor_df['dwell_time'].std():.1f} ms")
            print(f"  Std flight time:  {impostor_df['flight_time'].std():.1f} ms")
    
    def plot_all(self):
        """Generate all visualizations"""
        
        genuine_df = self.df[self.df['user_type'] == 'genuine']
        impostor_df = self.df[self.df['user_type'] == 'impostor']
        
        # 1. Genuine users detailed analysis
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Dwell time distribution (genuine users)
        ax = axes[0, 0]
        for user in genuine_df['user'].unique():
            user_data = genuine_df[genuine_df['user'] == user]['dwell_time']
            ax.hist(user_data, bins=30, alpha=0.6, label=user.title(), edgecolor='black')
        ax.set_xlabel('Dwell Time (ms)', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('Dwell Time Distribution (Genuine Users)', fontweight='bold', fontsize=12)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Plot 2: Flight time distribution (genuine users)
        ax = axes[0, 1]
        for user in genuine_df['user'].unique():
            user_data = genuine_df[genuine_df['user'] == user]['flight_time']
            user_data = user_data[user_data > 0]
            ax.hist(user_data, bins=30, alpha=0.6, label=user.title(), edgecolor='black')
        ax.set_xlabel('Flight Time (ms)', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('Flight Time Distribution (Genuine Users)', fontweight='bold', fontsize=12)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Plot 3: Box plot - Dwell time (genuine users)
        ax = axes[1, 0]
        dwell_data = [genuine_df[genuine_df['user'] == user]['dwell_time'] 
                      for user in genuine_df['user'].unique()]
        bp = ax.boxplot(dwell_data, labels=[u.title() for u in genuine_df['user'].unique()],
                       patch_artist=True)
        for patch, color in zip(bp['boxes'], sns.color_palette("Set2")):
            patch.set_facecolor(color)
        ax.set_ylabel('Dwell Time (ms)', fontweight='bold')
        ax.set_xlabel('User', fontweight='bold')
        ax.set_title('Dwell Time Consistency (Genuine Users)', fontweight='bold', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        # Plot 4: Box plot - Flight time (genuine users)
        ax = axes[1, 1]
        flight_data = [genuine_df[(genuine_df['user'] == user) & (genuine_df['flight_time'] > 0)]['flight_time']
                       for user in genuine_df['user'].unique()]
        bp = ax.boxplot(flight_data, labels=[u.title() for u in genuine_df['user'].unique()],
                       patch_artist=True)
        for patch, color in zip(bp['boxes'], sns.color_palette("Set2")):
            patch.set_facecolor(color)
        ax.set_ylabel('Flight Time (ms)', fontweight='bold')
        ax.set_xlabel('User', fontweight='bold')
        ax.set_title('Flight Time Consistency (Genuine Users)', fontweight='bold', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('genuine_users_stats.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved: genuine_users_stats.png")
        plt.show()
    
        # 2. Genuine vs Impostor comparison
        if len(impostor_df) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Plot 1: Dwell time comparison
            ax = axes[0]
            ax.hist(genuine_df['dwell_time'], bins=40, alpha=0.6, 
                   label='Genuine Users', color='green', edgecolor='black')
            ax.hist(impostor_df['dwell_time'], bins=40, alpha=0.6, 
                   label='Impostors', color='red', edgecolor='black')
            ax.set_xlabel('Dwell Time (ms)', fontweight='bold')
            ax.set_ylabel('Frequency', fontweight='bold')
            ax.set_title('Dwell Time: Genuine vs Impostors', fontweight='bold', fontsize=12)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            # Plot 2: Flight time comparison
            ax = axes[1]
            genuine_flight = genuine_df[genuine_df['flight_time'] > 0]['flight_time']
            impostor_flight = impostor_df[impostor_df['flight_time'] > 0]['flight_time']
            ax.hist(genuine_flight, bins=40, alpha=0.6, 
                   label='Genuine Users', color='green', edgecolor='black')
            ax.hist(impostor_flight, bins=40, alpha=0.6, 
                   label='Impostors', color='red', edgecolor='black')
            ax.set_xlabel('Flight Time (ms)', fontweight='bold')
            ax.set_ylabel('Frequency', fontweight='bold')
            ax.set_title('Flight Time: Genuine vs Impostors', fontweight='bold', fontsize=12)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('genuine_vs_impostor.png', dpi=300, bbox_inches='tight')
            print("✓ Saved: genuine_vs_impostor.png")
            plt.show()
        
        # 3. Typing Speed Analysis
        speed_df = self.calculate_typing_speed()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Speed distribution by user type
        ax = axes[0]
        genuine_speeds = speed_df[speed_df['user_type'] == 'genuine']
        impostor_speeds = speed_df[speed_df['user_type'] == 'impostor']
        
        if len(genuine_speeds) > 0:
            ax.hist(genuine_speeds['typing_speed'], bins=30, alpha=0.6, 
                   label='Genuine Users', color='green', edgecolor='black')
        if len(impostor_speeds) > 0:
            ax.hist(impostor_speeds['typing_speed'], bins=30, alpha=0.6, 
                   label='Impostors', color='red', edgecolor='black')
        
        ax.set_xlabel('Typing Speed (chars/sec)', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('Typing Speed Distribution', fontweight='bold', fontsize=12)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Plot 2: Average metrics comparison
        ax = axes[1]
        metrics = ['Dwell Time\n(ms)', 'Flight Time\n(ms)']
        genuine_means = [genuine_df['dwell_time'].mean(), 
                        genuine_df[genuine_df['flight_time'] > 0]['flight_time'].mean()]
        impostor_means = [impostor_df['dwell_time'].mean(), 
                         impostor_df[impostor_df['flight_time'] > 0]['flight_time'].mean()] if len(impostor_df) > 0 else [0, 0]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax.bar(x - width/2, genuine_means, width, label='Genuine', color='green', alpha=0.7)
        if len(impostor_df) > 0:
            ax.bar(x + width/2, impostor_means, width, label='Impostor', color='red', alpha=0.7)
        
        ax.set_ylabel('Time (ms)', fontweight='bold')
        ax.set_title('Average Timing Metrics', fontweight='bold', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('typing_metrics_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: typing_metrics_comparison.png")
        plt.show()


# ============================================================
# USAGE
# ============================================================
if __name__ == "__main__":
    # Initialize with both genuine and impostor folders
    viz = SimpleKeystrokeVisualizer(
        genuine_folder="data/genuine",
        impostor_folder="data/impostors"
    )

    # Load data
    viz.load_data()
    
    # Print statistics
    viz.print_summary_stats()
    
    # Generate all plots
    viz.plot_all()
    
    print("\n" + "="*60)
    print("✓ DONE! Check the generated PNG files.")
    print("="*60)