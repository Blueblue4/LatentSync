# 1 Performance
## Setup
Machine: Ubuntu 20.04, 1x RTX 3090 CUDA 12.1, 
Run setup as before (note the updated requirements.txt)
Run inference with profiling:
```
python -m scripts.inference \
--unet_config_path "configs/unet/stage2.yaml" \
--inference_ckpt_path "checkpoints/old/latentsync_unet.pt" \
--inference_steps 20 \
--guidance_scale 1.5 \
--video_path "assets/demo1_video.mp4" \
--audio_path "assets/demo1_audio.wav" \
--video_out_path "video_out_lq.mp4" \
--precompute
```
### Timeline
Notes 242 Frames
- Load masks 3s
- Audio prep 1.5s (parallel)
- Video preprocessing 15.
	- 9s initial, last 5.5s parallel cpu and some cuda
- Diffusion loop ~36s 16x 2.4s (last one shorter)
	- prep mask 105ms
	- prep img loop 105ms
	- diffusion 1.9s
	- decode latents 247ms
- Resore video 6.3 s
- save 7.2s
- join audio and video 5.5s
open `trace.json` in [https://ui.perfetto.dev/](https://ui.perfetto.dev/) 
## Pipeline Improvements 
### Flash attention
Issue on Ubuntu 20.04 [gh](https://github.com/Dao-AILab/flash-attention/issues/1708):
```
import flash_attn_2_cuda as flash_attn_gpu
ImportError: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.32'
```
Fix:
Using flash attention flash_attn==2.7.4.post1	
newest flash attention updates likely not relevant for 3090

Result: 
Negible on a 3090, likely better on newer/server GPUs

### Torch compile
Note that for deployment we would add torch.export and AOTI, since we don't want to compile the model on each startup.

issue1: doesn't work with deep cache enabled (needs a workaround -- not implemented)
Fix: deactivate deep_cache 
issue2:  typecasting  unet.py:381
// debug with env_var TORCH_LOGS=+dynamo;TORCHDYNAMO_VERBOSE=1
```Python
t_emb = t_emb.to(dtype=torch.float16)
# dirty fix
t_emb = t_emb.to(dtype=self.dtype)
```
Baseline: 103s 
Improved: 93s (Likely better with more modern GPU)

### Precompute and cache video-(masks, mask-features, etc.)
Since the application likely only uses one or a small fixed number of video sources, it makes sense to cache their preprocessing and computations (including the masks- and image- latents).
see LipsyncPipeline.precompute_video_frames and inference loop in lipsync_pipline.py:313 

Baseline: 103s 
Improved: 91s (likely faster if features are pre-loaded earlier)
### Further general improvements (Not implemented) 
tensorRT/ONNX, parallelized CPU processing, closer look at CPU-GPU data transfer

#  2 Improvements
Some identified issues:  

- Visual artifacts
- Repetition of movements in longer videos
- gestures not fitting to narrative
- Video ending mid-gesture 
- Additional artifacts/changes for extreme sounds/mimicry (wide open mouth)

## Training/Architecture

#### Adversarial loss/training
- Pro:
Reduce artifacts and grainy-ness
Sharper frames
- Con:
hard to balance training
somewhat overlaps with the objective of SyncNet supervision

#### Train on higher resolution 
Better supervision in image space -> fewer visual artifacts
Optionally distill into lower resolution model for faster inference. 

#### Vary the mask size during training 
Reduces artifacts at face-edges (also helps in portrait animation)

## General
#### Multiple reference videos
For each ID record multiple videos of different length. Use video with matching length for each speech input to ensure proper "ending" for the individual clips.  
#### Longer videos
Track the position in a longer video for longer conversations and optionally diffuse the video to a neutral pose at the end of each speech snippet.


# 3 Deployment
## System Architecture
The video stream needs to be streamed in near real-time. The model needs about 9GB VRAM. RAM requirements ~ 

### Inference Infrastructure
- GPU nodes with NVIDIA T4/A10 instances (16GB VRAM minimum)
- CPU instances with 32GB RAM (2x headroom for 15GB requirement)
- Load balancer with sticky sessions for multi-request conversations
- Request queue with priority scheduling for fairness

### Latency Budget Allocation
- Network ingress: 10-50ms
- Queue wait: 0-500ms (P95)
- Model inference: 200-800ms per frame
- Video encoding: 50-100ms
- Network egress: 10-50ms
- Total target: <2s for first frame, <100ms inter-frame

## Deployment Strategy

### Scaling Configuration
- Horizontal pod autoscaling based on GPU utilization (target 70%)
- Pre-warmed model instances to avoid cold start (500ms savings)
- Regional deployment with GeoDNS routing

### Memory Management
- Model weights pinned in GPU memory
- CPU memory: 15GB allocated (10GB model + 5GB buffer)
- Shared memory for inter-process communication
- Memory leak detection with automatic pod recycling at 90% usage

## Failure Modes and Mitigation

### Primary Failure Scenarios
1. **GPU OOM**: Implement request admission control, reject when >80% VRAM
2. **Model inference timeout**: Circuit breaker at 5s, fallback to static avatar
3. **Queue overflow**: Exponential backoff, client-side retry with jitter
4. **Corrupted output**: Validation layer checking frame coherence

### Graceful Degradation Path
- **Tier 1**: Full quality real-time generation
- **Tier 2**: Reduced resolution/framerate
- **Tier 3**: Static image with text response

## Monitoring and Observability

### Key Metrics
- **Latency**: P50/P95/P99 for each pipeline stage
- **Throughput**: Requests/second, frames/second per GPU
- **Resource Utilization**: GPU/CPU/Memory usage per pod
- **Error Rates**: Timeout/OOM/validation failures by type
- **Queue Depth**: Current size and wait time distribution

### Alerting Thresholds
- P95 latency >3s: Page on-call
- GPU utilization >85% for 5 min: Scale up
- Error rate >1%: Investigate
- Queue depth >100: Capacity alert
- Memory usage >85%: Pod recycling warning

### Logging Strategy
- Request tracing with correlation IDs
- Model inference profiling (frame generation time)
- Input/output samples for quality debugging
- Resource allocation events

## Capacity Planning

### Initial Deployment
- 10 GPU nodes per region (3 regions)
- 50 concurrent users per GPU
- 1500 peak concurrent users globally
- 20% headroom for traffic spikes

### Scaling Triggers
- Add node when avg GPU util >70% for 2 min
- Remove node when avg GPU util <40% for 10 min
- Regional burst capacity: 2x baseline
- Global failover capacity: 1.5x single region

