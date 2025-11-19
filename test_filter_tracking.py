"""
Enhanced test with detailed filter_fn internal tracking
"""
import sys
sys.path.insert(0, '/workspace/codes/verl')

from verl.utils.database.mysql import MySQLRolloutORM
import pandas as pd


def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def detailed_filter_analysis(run_id="results/test_1115", top_n=2, rollout_n=8):
    """
    Detailed step-by-step analysis of why filter_fn returns empty
    """
    print_section("STEP 1: Fetch Raw Data")
    
    db_manager = MySQLRolloutORM(
        config={
            'host': '172.16.0.2',
            'user': 'dart_rollouter',
            'password': 'Dt8@Rx9p',
            'database': 'dart_database',
            'port': 3306,
            'charset': 'utf8mb4'
        }
    )
    
    data = db_manager.get_rollouts_by_run_id(run_id=run_id)
    print(f"\n  ✅ Total raw data: {len(data)} records")
    
    if not data:
        print("  ❌ No data found!")
        return
    
    df = pd.DataFrame(data)
    
    # Analyze initial data
    print(f"\n  Raw Data Overview:")
    print(f"    - Columns: {list(df.columns)}")
    print(f"    - Shape: {df.shape}")
    
    print_section("STEP 2: Get top_mvs (Latest Checkpoint Paths)")
    
    top_mvs = db_manager.get_latest_n_checkpoint_paths(run_id=run_id, n=top_n)
    print(f"\n  ✅ Got {len(top_mvs)} checkpoint paths:")
    for i, path in enumerate(top_mvs, 1):
        print(f"    {i}. {path}")
    
    print_section("STEP 3: Manual Filter Step-by-Step Analysis")
    
    # Step 3.1: Check model_version in top_mvs
    print(f"\n  Filter Step 3.1: model_version ∈ top_mvs")
    print(f"    - top_mvs = {top_mvs}")
    print(f"\n    - Unique model_versions in data:")
    mv_counts = df['model_version'].value_counts()
    for mv, count in mv_counts.items():
        in_top = "✅ IN top_mvs" if mv in top_mvs else "❌ NOT in top_mvs"
        print(f"      {mv}: {count} records  {in_top}")
    
    df_step1 = df[df['model_version'].isin(top_mvs)].copy()
    print(f"\n    📊 After model_version filter: {len(df_step1)} / {len(df)} records ({100*len(df_step1)/len(df):.1f}%)")
    
    if df_step1.empty:
        print(f"\n  ❌ PROBLEM FOUND: No records match top_mvs!")
        print(f"  💡 This means the model_versions in your data don't match the checkpoint paths.")
        return
    
    # Step 3.2: Check used == 0
    print(f"\n  Filter Step 3.2: used == 0")
    print(f"\n    - Used value distribution in step1 data:")
    used_counts = df_step1['used'].value_counts()
    for used_val, count in used_counts.items():
        status = "✅ PASS" if used_val == 0 else "❌ FAIL"
        print(f"      used={used_val}: {count} records  {status}")
    
    df_step2 = df_step1[df_step1['used'] == 0].copy()
    print(f"\n    📊 After used==0 filter: {len(df_step2)} / {len(df_step1)} records ({100*len(df_step2)/len(df_step1):.1f}% of step1)")
    
    if df_step2.empty:
        print(f"\n  ❌ PROBLEM FOUND: All records have used != 0!")
        print(f"  💡 All matching model_version data has already been used for training.")
        return
    
    # Step 3.3: Check valid reward
    print(f"\n  Filter Step 3.3: valid reward (not -1, not NaN)")
    df_step3 = df_step2.copy()
    df_step3['reward'] = pd.to_numeric(df_step3['reward'], errors='coerce')
    
    print(f"\n    - Reward distribution in step2 data:")
    reward_before = df_step2['reward'].value_counts().sort_index()
    for reward_val, count in reward_before.items():
        status = "✅ VALID" if (pd.notna(reward_val) and reward_val != -1) else "❌ INVALID"
        print(f"      reward={reward_val}: {count} records  {status}")
    
    df_step3 = df_step3.loc[df_step3['reward'].notna() & (df_step3['reward'] != -1)]
    print(f"\n    📊 After valid reward filter: {len(df_step3)} / {len(df_step2)} records ({100*len(df_step3)/len(df_step2):.1f}% of step2)")
    
    if df_step3.empty:
        print(f"\n  ❌ PROBLEM FOUND: All rewards are either -1 or NaN!")
        return
    
    # Step 3.4: Check per-task eligibility
    print(f"\n  Filter Step 3.4: Per-task eligibility")
    print(f"    Requirements:")
    print(f"      - cnt >= {rollout_n} (samples per task)")
    print(f"      - mean_reward ∈ [0, 1.0)")
    
    grp = df_step3.groupby('task_id').agg(
        cnt=('trajectory_id', 'size'),
        mean_reward=('reward', 'mean')
    )
    
    print(f"\n    - Task-level statistics:")
    for task_id, row in grp.iterrows():
        cnt = row['cnt']
        mean_r = row['mean_reward']
        
        cnt_pass = "✅" if cnt >= rollout_n else f"❌ (need {rollout_n})"
        mean_pass = "✅" if (0 <= mean_r < 1.0) else f"❌ (mean={mean_r:.4f})"
        overall = "✅ ELIGIBLE" if (cnt >= rollout_n and 0 <= mean_r < 1.0) else "❌ REJECTED"
        
        print(f"      task_id={task_id}:")
        print(f"        cnt={cnt} {cnt_pass}")
        print(f"        mean_reward={mean_r:.4f} {mean_pass}")
        print(f"        → {overall}")
    
    eligible = grp[(grp['cnt'] >= rollout_n) & (grp['mean_reward'].ge(0) & grp['mean_reward'].lt(1.0))]
    df_step4 = df_step3[df_step3['task_id'].isin(eligible.index)]
    
    print(f"\n    📊 Eligible tasks: {len(eligible)} / {len(grp)} tasks")
    print(f"    📊 After eligibility filter: {len(df_step4)} / {len(df_step3)} records")
    
    if df_step4.empty:
        print(f"\n  ❌ PROBLEM FOUND: No tasks meet eligibility criteria!")
        print(f"\n  💡 Possible reasons:")
        print(f"     - Tasks don't have enough samples (need >= {rollout_n})")
        print(f"     - Task mean_reward is outside [0, 1.0) range")
        print(f"     - All tasks have mean_reward = 1.0 (all perfect, acc_max excludes them)")
        return
    
    print_section("SUMMARY: Root Cause Analysis")
    
    print(f"\n  Data flow through filter_fn:")
    print(f"    1. Raw data:              {len(df)} records")
    print(f"    2. After model_version:   {len(df_step1)} records ({100*len(df_step1)/len(df):.1f}%)")
    print(f"    3. After used==0:         {len(df_step2)} records ({100*len(df_step2)/len(df):.1f}%)")
    print(f"    4. After valid reward:    {len(df_step3)} records ({100*len(df_step3)/len(df):.1f}%)")
    print(f"    5. After eligibility:     {len(df_step4)} records ({100*len(df_step4)/len(df):.1f}%)")
    
    if len(df_step4) == 0:
        print(f"\n  ❌ FINAL RESULT: No data passed all filters!")
    else:
        print(f"\n  ✅ FINAL RESULT: {len(df_step4)} records passed all filters")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("  DETAILED FILTER_FN ANALYSIS")
    print("="*80)
    
    detailed_filter_analysis(
        run_id="results/test_1115",
        top_n=2,
        rollout_n=8
    )
    
    print("\n" + "="*80)
    print("  ANALYSIS COMPLETED")
    print("="*80 + "\n")