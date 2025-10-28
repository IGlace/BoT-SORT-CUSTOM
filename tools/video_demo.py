import sys
import os
import os.path as osp
import time
import cv2
import torch
import yaml
from types import SimpleNamespace
from loguru import logger

sys.path.append('.')

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from tracker.bot_sort import BoTSORT
from tracker.tracking_utils.timer import Timer


def load_yaml(file_path):
    """Load YAML configuration file and convert to namespace."""
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert nested dict to namespace for easier access
    def dict_to_namespace(d):
        namespace = SimpleNamespace()
        for key, value in d.items():
            if isinstance(value, dict):
                setattr(namespace, key, dict_to_namespace(value))
            else:
                setattr(namespace, key, value)
        return namespace
    
    return dict_to_namespace(config)


def build_tracker_args(config, device):
    """Map YAML config to the argument namespace expected by BoTSORT."""
    tracker_cfg = config.tracker
    reid_cfg = config.reid
    cmc_cfg = config.cmc

    fps = tracker_cfg.fps
    fuse_score = bool(tracker_cfg.fuse_score)
    with_reid = bool(reid_cfg.enabled)

    tracker_name = "personal_tracker"  # Simple default name

    if isinstance(device, torch.device):
        device_type = device.type
    else:
        device_type = str(device)
    device_value = "cuda" if str(device_type).startswith("cuda") else "cpu"

    return SimpleNamespace(
        fps=fps,
        track_high_thresh=tracker_cfg.track_high_thresh,
        track_low_thresh=tracker_cfg.track_low_thresh,
        new_track_thresh=tracker_cfg.new_track_thresh,
        track_buffer=tracker_cfg.track_buffer,
        match_thresh=tracker_cfg.match_thresh,
        aspect_ratio_thresh=tracker_cfg.aspect_ratio_thresh,
        min_box_area=tracker_cfg.min_box_area,
        fuse_score=fuse_score,
        proximity_thresh=reid_cfg.proximity_thresh,
        appearance_thresh=reid_cfg.appearance_thresh,
        with_reid=with_reid,
        fast_reid_config=reid_cfg.fast_reid_config,
        fast_reid_weights=reid_cfg.fast_reid_weights,
        reid_ambiguity_thresh=reid_cfg.reid_ambiguity_thresh,
        reid_overlap_thresh=reid_cfg.reid_overlap_thresh,
        reid_min_track_age=reid_cfg.reid_min_track_age,
        reid_early_collect_offset=reid_cfg.reid_early_collect_offset,
        # Persistent ReID parameters
        persistent_max_age_minutes=reid_cfg.persistent_max_age_minutes,
        persistent_max_identities=reid_cfg.persistent_max_identities,
        persistent_similarity_threshold=reid_cfg.persistent_similarity_threshold,
        # Safety monitoring parameters
        high_density_threshold=tracker_cfg.high_density_threshold,
        safety_monitoring=tracker_cfg.safety_monitoring,
        cmc_method=cmc_cfg.method,
        name=tracker_name,
        ablation=False,
        mot20=not fuse_score,
        device=device_value,
    )


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class VideoTracker(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device=torch.device("cpu"),
        fp16=False,
        display=True
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.display = display
        
        if trt_file is not None:
            from torch2trt import TRTModule
            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))
            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt
            
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
        return outputs, img_info


def track_video(config, exp, predictor, tracker_args, video_path=None, output_dir=None, display=True, save_video=True, save_txt=True):
    if video_path is None:
        cap = cv2.VideoCapture(config.source.camera_id)
        video_name = f"camera_{config.source.camera_id}"
        if not video_name.lower().endswith(".mp4"):
            video_name = f"{video_name}.mp4"
    else:
        cap = cv2.VideoCapture(video_path)
        video_name = os.path.basename(video_path)

    if not cap.isOpened():
        source_desc = video_path if video_path is not None else f"camera {config.source.camera_id}"
        cap.release()
        raise RuntimeError(f"Unable to open video source: {source_desc}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or exp.test_size[1])
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or exp.test_size[0])
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = tracker_args.fps

    save_result = (save_video or save_txt) and output_dir is not None
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) if save_result else None
    vis_folder = None
    save_folder = None
    vid_writer = None

    if save_result:
        os.makedirs(output_dir, exist_ok=True)
        vis_folder = osp.join(output_dir, "track_vis")
        os.makedirs(vis_folder, exist_ok=True)
        save_folder = osp.join(vis_folder, timestamp)
        os.makedirs(save_folder, exist_ok=True)

    if save_video and save_folder:
        save_video_path = osp.join(save_folder, video_name)
        logger.info(f"video save path is {save_video_path}")
        vid_writer = cv2.VideoWriter(
            save_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )
        if not vid_writer.isOpened():
            logger.warning(f"Failed to open video writer at {save_video_path}. Video will not be saved.")
            vid_writer.release()
            vid_writer = None

    tracker_runtime_args = SimpleNamespace(**vars(tracker_args))
    tracker = BoTSORT(tracker_runtime_args, frame_rate=tracker_args.fps)
    timer = Timer()
    frame_id = 0
    results = [] if save_txt and save_result else None

    logger.info("Starting video tracking...")
    while True:
        if frame_id % 20 == 0:
            logger.info(f'Processing frame {frame_id} ({1. / max(1e-5, timer.average_time):.2f} fps)')
            
        ret_val, frame = cap.read()
        if not ret_val:
            break

        outputs, img_info = predictor.inference(frame, timer)
        scale = min(exp.test_size[0] / float(img_info['height'], ), exp.test_size[1] / float(img_info['width']))

        if outputs[0] is not None:
            outputs = outputs[0].cpu().numpy()
            detections = outputs[:, :7]
            detections[:, :4] /= scale

            online_targets = tracker.update(detections, img_info["raw_img"])

            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > tracker_runtime_args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > tracker_runtime_args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    if results is not None:
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
            
            timer.toc()
            online_im = plot_tracking(
                img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
            )
        else:
            timer.toc()
            online_im = img_info['raw_img']

        if vid_writer is not None:
            vid_writer.write(online_im)
        
        if display:
            cv2.imshow("online_im", online_im)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break

        frame_id += 1

    if save_txt and vis_folder and timestamp:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            if results:
                f.writelines(results)
        logger.info(f"save results to {res_file}")

    # Log statistics
    total_time = timer.total_time
    avg_fps = frame_id / total_time if total_time > 0 else 0
    logger.info(f"Total tracking time: {total_time:.2f}s for {frame_id} frames ({avg_fps:.2f} fps)")
    if tracker_runtime_args.with_reid:
        logger.info(f"ReID stats: Used in {tracker.total_reid_frames} frames, total {tracker.total_reid_calls} ReID calls")

    cap.release()
    if vid_writer is not None:
        vid_writer.release()
    cv2.destroyAllWindows()


def main(config_path):
    config = load_yaml(config_path)
    model_name = config.model.name if hasattr(config.model, "name") else None
    exp = get_exp(config.model.exp_file, model_name)

    output_cfg = config.output
    experiment_name = exp.exp_name
    output_dir = output_cfg.dir if output_cfg.dir else osp.join(exp.output_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    if config.model.trt:
        config.model.device = "gpu"

    requested_device = str(config.model.device).lower()
    if requested_device == "gpu" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU execution.")
        requested_device = "cpu"
    device = torch.device("cuda" if requested_device == "gpu" else "cpu")
    config.model.device = "gpu" if device.type == "cuda" else "cpu"

    if config.model.fp16 and device.type == "cpu":
        logger.warning("FP16 inference is not supported on CPU. Disabling fp16.")
        config.model.fp16 = False

    tracker_args = build_tracker_args(config, device)

    logger.info(f"Running on device: {device}")
    logger.info(f"Tracker arguments: {tracker_args}")
    logger.info(f"Config loaded: {config}")

    # Allow optional overrides to fall back to experiment defaults when unset.
    model_conf = config.model.conf
    if model_conf is not None:
        exp.test_conf = model_conf
    model_nms = config.model.nms
    if model_nms is not None:
        exp.nmsthre = model_nms
    model_tsize = config.model.tsize
    if model_tsize is not None:
        exp.test_size = (model_tsize, model_tsize)

    model = exp.get_model().to(device)
    logger.info(f"Model loaded: {get_model_info(model, exp.test_size)}")
    model.eval()

    if not config.model.trt:
        if config.model.ckpt is None:
            ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
        else:
            ckpt_file = config.model.ckpt
        logger.info(f"Loading checkpoint: {ckpt_file}")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        logger.info("Checkpoint loaded successfully")

    if config.model.fuse:
        logger.info("Fusing model...")
        model = fuse_model(model)

    if config.model.fp16:
        model = model.half()

    if config.model.trt:
        assert not config.model.fuse, "TensorRT model does not support fusing!"
        trt_file = osp.join(output_dir, "model_trt.pth")
        assert osp.exists(trt_file), "TensorRT model not found! Run tools/trt.py first."
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
    else:
        trt_file = None
        decoder = None

    display_cfg = config.display
    display_enabled = bool(display_cfg.enabled)
    save_video = bool(output_cfg.save_video)
    save_txt = bool(output_cfg.save_txt)

    predictor = VideoTracker(
        model=model,
        exp=exp,
        trt_file=trt_file,
        decoder=decoder,
        device=device,
        fp16=config.model.fp16,
        display=display_enabled
    )

    source_cfg = config.source

    # Process video or camera stream
    if source_cfg.type == "video":
        track_video(config, exp, predictor, tracker_args, source_cfg.path, output_dir, display_enabled, save_video, save_txt)
    elif source_cfg.type == "camera":
        track_video(config, exp, predictor, tracker_args, None, output_dir, display_enabled, save_video, save_txt)
    else:
        raise ValueError(f"Unknown source type: {source_cfg.type}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("BoT-SORT Video Demo!")
    parser.add_argument("--config", default="./configs/default.yaml", required=False, help="Path to config file")
    args = parser.parse_args()

    if not args.config:
        args.config = "./configs/default.yaml"
    
    # Configure logger to show INFO level messages (required for safety monitoring visibility)
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
    
    main(args.config)
