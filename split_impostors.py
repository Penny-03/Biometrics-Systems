import json
import os

# --- Configuration ---
input_files = [
    'data/keystroke_data_jolaaa_1764869557315.json',
    'data/keystroke_data_xhorxhi_1764872745524.json'
]

# The valid characters allowed in the output
TARGET_PHRASE = "the quick brown fox jumps over the lazy dog"
ALLOWED_KEYS = set(TARGET_PHRASE) # Creates a set of characters: {'a', 'b', 'c', ..., ' '}

def split_and_clean_impostors(files):
    # Dictionary to hold data for each user
    # Structure: { 'userid': { 'metadata': ..., 'sessions': [] } }
    user_databases = {}

    print(f"Processing {len(files)} files...")

    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                source_data = json.load(f)
        except FileNotFoundError:
            print(f"Skipping {file_path} (not found)")
            continue

        # Get base metadata
        base_metadata = source_data.get('metadata', {})
        
        # Iterate through every session in the file
        for session in source_data.get('sessions', []):
            
            # Temporary storage to sort keystrokes by user within this session
            # Structure: { 'userid': [keystroke1, keystroke2, ...] }
            session_users_map = {}

            # 1. Iterate through data points to identify users and filter keys
            for keystroke in session.get('data', []):
                # Normalize User ID to lowercase (Small Caps)
                raw_uid = keystroke.get('userId') or session.get('userId') or base_metadata.get('userId')
                if not raw_uid: continue 
                
                uid = str(raw_uid).lower()
                
                # Check for "Garbage" characters
                key_char = keystroke.get('key')
                
                # FILTER: Delete if the key is not in the phrase
                if key_char not in ALLOWED_KEYS:
                    continue 

                # Initialize list for this user if new
                if uid not in session_users_map:
                    session_users_map[uid] = []
                
                # Update the keystroke object itself to have the lowercase ID
                keystroke['userId'] = uid
                session_users_map[uid].append(keystroke)

            # 2. Reconstruct sessions for each user found
            for uid, cleaned_data in session_users_map.items():
                if not cleaned_data: continue

                # Initialize user database entry if this is the first time seeing this user
                if uid not in user_databases:
                    new_metadata = base_metadata.copy()
                    new_metadata['userId'] = uid
                    user_databases[uid] = {
                        'metadata': new_metadata,
                        'sessions': []
                    }

                # Create a new clean session object
                new_session = session.copy()
                new_session['userId'] = uid
                new_session['data'] = cleaned_data
                new_session['keystrokeCount'] = len(cleaned_data) # Update count
                
                # Append to that user's list
                user_databases[uid]['sessions'].append(new_session)

    # 3. Save separate files with "impostor_" prefix
    generated_files = []
    for uid, data in user_databases.items():
        # Update total sessions count for the user
        data['metadata']['totalSessions'] = len(data['sessions'])
        
        # New filename format: impostor_userid.json
        output_filename = f"impostor_{uid}.json"
        
        with open(output_filename, 'w') as f:
            json.dump(data, f, indent=2)
        generated_files.append(output_filename)

    return generated_files

# --- Execution ---
created_files = split_and_clean_impostors(input_files)

print("\n" + "="*30)
print("IMPOSTOR SPLIT COMPLETE")
print("="*30)
print(f"Generated {len(created_files)} separate files:")
for f in created_files:
    print(f" - {f}")