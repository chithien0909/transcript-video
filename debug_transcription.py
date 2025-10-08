#!/usr/bin/env python3
"""
Debug script to test transcription process
"""

import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_transcription():
    """Test the transcription process step by step."""
    print("Testing transcription process...")
    
    # Test file path
    video_file = "videos/Tr_m_tin_sang_8_10_-_Ha_N_i_va_nhi_u_t_nh_phia_B_c_ti_p_t_c_m_a_l_n_Tri_u_c_ng_C_n_Th_v_t_bao_ng.mp4"
    
    if not os.path.exists(video_file):
        print(f"Video file not found: {video_file}")
        return False
    
    print(f"Video file found: {video_file}")
    
    # Test folder creation
    video_dir = os.path.dirname(video_file)
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder_name = f"{video_name}_transcription_{timestamp}"
    out_dir = os.path.join(video_dir, output_folder_name)
    os.makedirs(out_dir, exist_ok=True)
    
    base_name = os.path.join(out_dir, video_name)
    
    print(f"Output directory: {out_dir}")
    print(f"Base name: {base_name}")
    
    # Test file creation
    test_files = [
        base_name + ".srt",
        base_name + ".vtt", 
        base_name + ".json",
        base_name + ".txt"
    ]
    
    for test_file in test_files:
        try:
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write("Test content")
            print(f"Created test file: {test_file}")
            os.remove(test_file)  # Clean up
        except Exception as e:
            print(f"Failed to create {test_file}: {e}")
            return False
    
    print("File creation test passed!")
    return True

if __name__ == "__main__":
    test_transcription()
