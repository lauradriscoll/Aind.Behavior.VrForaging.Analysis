import numpy as np
import json
from pathlib import Path
import pandas as pd
from aind_vr_foraging_analysis.utils.parsing.data_access import load_session


def extract_window_data_from_session_by_odor(
    session_path: str,
    window_size: int = 100,
    step_size: int = 10,
    use_time: bool = False,
    exclude_odors: list = ['Amyl Acetate']
):
    """
    Extract sliding windows of behavioral data from a VR foraging session,
    SEPARATED BY ODOR TYPE into separate folders.
    
    Parameters:
    -----------
    session_path : str
        Full path to the session directory
    window_size : int, default=100
        Number of sites per window
    step_size : int, default=10
        Number of sites to slide window by
    use_time : bool, default=False
        If True, use time_since_entry; if False, use start_position
    exclude_odors : list, default=['Amyl Acetate']
        List of odor types to exclude
        
    Returns:
    --------
    base_output_dir : Path
        Path to the base directory containing odor-specific subdirectories
    """
    
    session_path = Path(session_path)
    
    # Load session data
    print(f"Loading session from: {session_path}")
    all_epochs, stream_data, raw_data = load_session(session_path)
    print(f"Loaded {len(all_epochs)} total epochs")
    
    # Filter for OdorSites
    odor_sites = all_epochs[all_epochs['label'] == 'OdorSite'].copy()
    n_odor_sites = len(odor_sites)
    print(f"Found {n_odor_sites} OdorSites")
    
    # Get unique odor types
    unique_odors = odor_sites['patch_label'].unique()
    print(f"Odor types: {unique_odors}")
    
    # Filter out excluded odors
    odors_to_process = [o for o in unique_odors if o not in exclude_odors]
    print(f"Processing odors: {odors_to_process}")
    print(f"Excluding odors: {exclude_odors}")
    
    # Create base output directory
    base_output_dir = session_path / f'{window_size}_window_data_by_odor'
    base_output_dir.mkdir(exist_ok=True)
    
    # Process each odor type separately
    odor_summary = {}
    
    for odor_type in odors_to_process:
        print(f"\n{'='*70}")
        print(f"Processing odor type: {odor_type}")
        print(f"{'='*70}")
        
        # Create odor-specific directory
        odor_safe = odor_type.replace(' ', '_').replace('-', '_')
        odor_dir = base_output_dir / odor_safe
        odor_dir.mkdir(exist_ok=True)
        
        # Filter for this odor type
        odor_sites_filtered = odor_sites[odor_sites['patch_label'] == odor_type].copy()
        n_sites = len(odor_sites_filtered)
        print(f"  {n_sites} sites for {odor_type}")
        
        if n_sites < window_size:
            print(f"Not enough sites for {odor_type}, skipping")
            continue
        
        # Compute patch-relative positions
        odor_sites_filtered = odor_sites_filtered.reset_index(drop=True)
        patch_relative_positions = []
        patch_relative_times = []
        
        for i, row in odor_sites_filtered.iterrows():
            patch_num = row['patch_number']
            
            # Find patch entry position
            patch_epochs = all_epochs[all_epochs['patch_number'] == patch_num]
            intersites = patch_epochs[patch_epochs['label'] == 'InterSite']
            
            if len(intersites) > 0:
                patch_entry_position = intersites.iloc[0]['start_position']
            else:
                first_odor = patch_epochs[patch_epochs['label'] == 'OdorSite'].iloc[0]
                patch_entry_position = first_odor['start_position']
            
            patch_relative_position = row['start_position'] - patch_entry_position
            patch_relative_time = row['time_since_entry']
            
            patch_relative_positions.append(patch_relative_position)
            patch_relative_times.append(patch_relative_time)
        
        odor_sites_filtered['patch_relative_position'] = patch_relative_positions
        odor_sites_filtered['patch_relative_time'] = patch_relative_times
        
        # Create windows for this odor
        n_windows = (n_sites - window_size) // step_size + 1
        print(f"  Creating {n_windows} windows")
        
        windows_saved = []
        
        for window_idx in range(n_windows):
            start_idx = window_idx * step_size
            end_idx = start_idx + window_size
            
            window_sites = odor_sites_filtered.iloc[start_idx:end_idx]
            
            # Extract data
            window_data = np.zeros((window_size, 3))
            
            for i, (_, site) in enumerate(window_sites.iterrows()):
                patch_metric = site['patch_relative_time'] if use_time else site['patch_relative_position']
                reward = 1 if site['is_reward'] else 0
                stopped = 1 if site['is_choice'] else 0
                
                window_data[i] = [patch_metric, reward, stopped]
            
            # Save in odor-specific directory
            window_filename = f"window_{window_idx:03d}.npy"
            window_path = odor_dir / window_filename
            np.save(window_path, window_data)
            windows_saved.append(window_filename)
        
        # Save metadata for this odor in its directory
        odor_metadata = {
            'odor_type': odor_type,
            'odor_safe_name': odor_safe,
            'n_sites': n_sites,
            'n_windows': n_windows,
            'window_size': window_size,
            'step_size': step_size,
            'use_time': use_time,
            'patch_metric': 'patch_relative_time' if use_time else 'patch_relative_position',
            'patch_entry_method': 'first_InterSite',
            'windows_saved': windows_saved
        }
        
        odor_metadata_path = odor_dir / 'metadata.json'
        with open(odor_metadata_path, 'w') as f:
            json.dump(odor_metadata, f, indent=2)
        
        print(f"  ✓ Saved {len(windows_saved)} windows to {odor_dir}")
        
        # Add to summary
        odor_summary[odor_type] = {
            'odor_safe_name': odor_safe,
            'n_sites': n_sites,
            'n_windows': n_windows,
            'directory': str(odor_dir.relative_to(session_path))
        }
    
    # Save session-level metadata
    session_metadata = {
        'session_path': str(session_path),
        'mouse_id': session_path.parent.name,
        'timestamp': session_path.name.split('_')[-1],
        'window_size': window_size,
        'step_size': step_size,
        'use_time': use_time,
        'excluded_odors': exclude_odors,
        'odor_summary': odor_summary
    }
    
    session_metadata_path = base_output_dir / 'session_metadata.json'
    with open(session_metadata_path, 'w') as f:
        json.dump(session_metadata, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Session metadata saved to {session_metadata_path}")
    print(f"{'='*70}")
    
    return base_output_dir


def batch_process_sessions_by_odor(
    base_path: str,
    window_size: int = 100,
    step_size: int = 10,
    use_time: bool = False,
    exclude_odors: list = ['Amyl Acetate'],
    skip_existing: bool = False
):
    """
    Batch process all sessions, extracting windows by odor type.
    """
    base_path = Path(base_path)
    
    # Find all mouse directories
    mouse_dirs = [d for d in base_path.iterdir() if d.is_dir()]
    print(f"Found {len(mouse_dirs)} mouse directories")
    
    # Collect all sessions
    all_sessions = []
    for mouse_dir in sorted(mouse_dirs):
        mouse_id = mouse_dir.name
        session_dirs = [d for d in mouse_dir.iterdir() if d.is_dir()]
        
        for session_dir in sorted(session_dirs):
            session_name = session_dir.name
            timestamp = session_name.split('_')[-1] if '_' in session_name else 'unknown'
            
            has_behavior = (session_dir / "behavior").exists()
            has_window_data = (session_dir / "window_data_by_odor").exists()
            
            all_sessions.append({
                'mouse_id': mouse_id,
                'session_dir': session_dir,
                'timestamp': timestamp,
                'has_behavior': has_behavior,
                'has_window_data': has_window_data,
                'status': 'not_started',
                'error': None
            })
    
    sessions_df = pd.DataFrame(all_sessions)
    
    print(f"\n{'='*70}")
    print(f"TOTAL: {len(sessions_df)} sessions found")
    print(f"  With behavior folder: {sessions_df['has_behavior'].sum()}")
    print(f"  Already have window_data_by_odor: {sessions_df['has_window_data'].sum()}")
    
    # Determine which to process
    if skip_existing:
        to_process = sessions_df[~sessions_df['has_window_data'] & sessions_df['has_behavior']]
    else:
        to_process = sessions_df[sessions_df['has_behavior']]
    
    print(f"  To process: {len(to_process)}")
    print(f"\n{'='*70}")
    
    # Process each session
    for idx, row in to_process.iterrows():
        print(f"\nProcessing [{idx+1}/{len(to_process)}]: {row['mouse_id']}/{row['timestamp']}")
        print("-" * 70)
        
        try:
            output_dir = extract_window_data_from_session_by_odor(
                session_path=str(row['session_dir']),
                window_size=window_size,
                step_size=step_size,
                use_time=use_time,
                exclude_odors=exclude_odors
            )
            sessions_df.at[idx, 'status'] = 'success'
            sessions_df.at[idx, 'has_window_data'] = True
            print(f"✓ Success")
            
        except Exception as e:
            sessions_df.at[idx, 'status'] = 'failed'
            sessions_df.at[idx, 'error'] = str(e)
            print(f"✗ Failed: {e}")
    
    # Summary
    print(f"\n{'='*70}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*70}")
    print(f"Total sessions: {len(to_process)}")
    print(f"Successful: {(sessions_df['status'] == 'success').sum()}")
    print(f"Failed: {(sessions_df['status'] == 'failed').sum()}")
    
    return sessions_df


# Main execution
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        base_path = "/Users/laura.driscoll/Documents/data/VR foraging/vr_foraging_data"
        
        print("BATCH PROCESSING MODE - BY ODOR TYPE")
        print(f"Base path: {base_path}")
        
        results_df = batch_process_sessions_by_odor(
            base_path=base_path,
            window_size=100,
            step_size=10,
            use_time=False,
            exclude_odors=['Amyl Acetate'],
            skip_existing=False
        )
        
        # Save results
        output_file = Path(base_path) / "batch_processing_by_odor_results.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
        
    else:
        # Single session test
        session_path = "/Users/laura.driscoll/Documents/data/VR foraging/vr_foraging_data/745305/745305_20241212T105857"
        
        output_dir = extract_window_data_from_session_by_odor(
            session_path=session_path,
            window_size=100,
            step_size=10,
            use_time=False,
            exclude_odors=['Amyl Acetate']
        )
        
        print(f"\nDone! Odor-separated windows saved to: {output_dir}")