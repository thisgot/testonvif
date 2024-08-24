import subprocess
import numpy as np
import cv2
import time

rtsp_url = 'rtsp://admin:123456qq@192.168.1.100:554/onvif1'
ffmpeg_path = './ffmpeg-m/bin/ffmpeg.exe'  # Atualize o caminho, se necessário

def encode_frame(frame):
    # Define FFmpeg command to encode a single frame to H.264
    command_encode = [
        ffmpeg_path,
        '-f', 'rawvideo',         # Input format
        '-pix_fmt', 'bgr24',      # Pixel format
        '-s', f'{frame.shape[1]}x{frame.shape[0]}',  # Frame size
        '-i', '-',                # Input from stdin
        '-c:v', 'libx264',        # H.264 encoding
        '-f', 'h264',             # Output format
        '-'                       # Output to stdout
    ]

    # Start FFmpeg process for encoding
    process = subprocess.Popen(command_encode, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Encode the frame
    encode_start_time = time.time()
    encoded_data, _ = process.communicate(input=frame.tobytes())
    encode_time = time.time() - encode_start_time
    print(f"Time to encode frame: {encode_time:.4f} s")

    return encoded_data

def decode_frame(encoded_data, width, height):
    # Define FFmpeg command to decode the H.264 encoded frame
    command_decode = [
        ffmpeg_path,
        '-f', 'h264',             # Input format
        '-i', '-',                # Input from stdin
        '-f', 'rawvideo',         # Output format
        '-pix_fmt', 'bgr24',      # Pixel format
        '-s', f'{width}x{height}', # Frame size
        '-'
    ]

    # Start FFmpeg process for decoding
    process = subprocess.Popen(command_decode, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Decode the frame
    decode_start_time = time.time()
    decoded_data = process.communicate(input=encoded_data)[0]
    decode_time = time.time() - decode_start_time
    print(f"Time to decode frame: {decode_time:.4f} s")

    # Convert decoded data to numpy array
    frame_size = width * height * 3  # Number of bytes per frame (for BGR24 format)
    decoded_frame = np.frombuffer(decoded_data, np.uint8).reshape((height, width, 3))

    return decoded_frame

def adjust_brightness(image, threshold=40, new_min=80, new_max=255): #make simple if < x then set to range new -max
    # Convert image to float and normalize to range [0, 1]
    image = image.astype(float) / 255.0
    
    # Define the threshold in normalized range
    threshold_norm = threshold / 255.0
    
    # Apply piecewise adjustment
    # Map values below threshold to the range [0, new_max - new_min]
    below_threshold = image < threshold_norm
    adjusted_image = np.where(below_threshold,
                              np.interp(image, [0, threshold_norm], [0, (new_max - new_min) / 255.0]),
                              image)
    
    # Map values above threshold to the range [new_min, 255]
    #adjusted_image = np.where(~below_threshold,
    #                          np.interp(image, [threshold_norm, 1], [new_min / 255.0, 1]),
    #                          adjusted_image)
    
    # Scale back to range [0, 255] and convert to uint8
    adjusted_image = np.clip(255 * adjusted_image, 0, 255).astype(np.uint8)
    
    return adjusted_image
    
    return adjusted_image

def adjust_gamma_1(image, gamma=1.0):
    # Convert image to float and normalize to range [0, 1]
    image = image.astype(float) / 255.0
    
    # Apply Gamma correction
    corrected = np.power(image, gamma)
    
    # Scale back to range [0, 255] and convert to uint8
    corrected = np.array(255 * corrected, dtype='uint8')
    
    return corrected
    
def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

def capture_frame_from_rtsp(rtsp_url):
    start_time = time.time()

    # Comando FFmpeg para capturar frames
    command_get_stream_n_decode = [
        ffmpeg_path,
        '-rtsp_transport', 'udp',  # ou 'tcp'
        '-i', rtsp_url,
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-'
    ]
    command_get_stream = [
        ffmpeg_path,
        '-rtsp_transport', 'udp',  # or 'tcp'
        '-i', rtsp_url,
        '-c', 'copy',  # Copy the stream without decoding
        '-c:v', 'rawvideo',
        '-f', 'rawvideo',  # Output format
        '-'
    ]

    command_verbose = [
        ffmpeg_path,
        '-rtsp_transport', 'udp',
        '-i', rtsp_url,
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-acodec', 'copy',
        '-'
    ]

    frame_size = 2304 * 1296 * 3  # Ajuste se necessário
    buffer_size = 1  # Tamanho do buffer
    print('t')
    # Iniciar o processo FFmpeg
    process = subprocess.Popen(command_get_stream_n_decode, stdout=subprocess.PIPE, bufsize=buffer_size)
    #process = subprocess.Popen(command_get_stream, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**6)
    print('t2')
    # Ler o primeiro frame para determinar a resolução
    raw_frame = process.stdout.read(frame_size)
    #raw_frame = process.stdout.read(1024)
    print('t3')
    if not raw_frame:
        print("Erro: Falha ao capturar o frame.")
        process.terminate()
        return
    
    # Calcular a resolução do frame
    width = 2304
    height = 1296
    
    print(f"Resolução detectada: {width}x{height}")

    frame_count = 0
    gamma = 2
    while True:
        frame_start_time = time.time()
        
        # Read the next frame from stdout
        read_start_time = time.time()
        raw_frame = process.stdout.read(width * height * 3)
        read_time = time.time() - read_start_time
        if not raw_frame:
            break
        
        try:
            # Process raw frame
            raw_start_time = time.time()
            frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))
            raw_time = time.time() - raw_start_time
            
            # Display frame
            display_start_time = time.time()
            cv2.imshow('RTSP Stream', frame)
            display_time = time.time() - display_start_time
            
            # Handle user input
            cv2wait_start_time = time.time()
            key = cv2.waitKey(1) & 0xFF
            if key == ord('+'): 
                gamma += 0.1
            elif key == ord('-'): 
                gamma -= 0.1
                gamma = max(gamma, 0.1)
            elif key == ord('q'): 
                break
            cv2_time = time.time() - cv2wait_start_time
        except Exception as e:
            print(f"Error processing frame: {e}")
            continue
        
        # Print timing information
        frame_time = time.time() - frame_start_time
        print(f"Frame {frame_count % 10} - Time to read frame: {read_time:.4f} s")
        print(f"Time to display frame: {display_time:.4f} s")
        print(f"Time for cv2 wait: {cv2_time:.4f} s")
        print(f"Total time for frame: {frame_time:.4f} s")
        frame_count += 1
        #encoded_data = encode_frame(frame)
        #decoded_frame = decode_frame(encoded_data, width, height)
        print("-" * 40)

    process.terminate()
    cv2.destroyAllWindows()

    total_time = time.time() - start_time
    print(f"Total time: {total_time:.4f} seconds")

capture_frame_from_rtsp(rtsp_url)
