"""
GenEval benchmark evaluation for VILA-U model
Adapted from HART implementation for image generation evaluation
"""
import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings("ignore")

try:
    from mmdet.apis import inference_detector, init_detector
    from mmdet import __version__ as mmdet_version
    MMDET_AVAILABLE = True
except ImportError:
    MMDET_AVAILABLE = False
    print("Warning: MMDetection not available. Object detection evaluation will be skipped.")

import vila_u
from utils import tracker


class GenEvalEvaluator:
    """
    GenEval evaluator for compositional text-to-image generation
    Evaluates object detection, counting, spatial relationships, and attributes
    """
    
    def __init__(self, model_path=None, config_path=None, device='cuda'):
        self.device = device
        self.detector = None
        
        if MMDET_AVAILABLE and model_path and config_path:
            try:
                self.detector = init_detector(config_path, model_path, device=device)
                print(f"Loaded MMDetection model: {config_path}")
            except Exception as e:
                print(f"Failed to load detector: {e}")
                self.detector = None
    
    def detect_objects(self, image_path, conf_threshold=0.3):
        """
        Detect objects in an image using MMDetection
        """
        if not self.detector:
            return []
        
        try:
            # Load image
            if isinstance(image_path, str):
                image = cv2.imread(image_path)
            else:
                image = np.array(image_path)
            
            # Run inference
            result = inference_detector(self.detector, image)
            
            # Extract detected objects
            detections = []
            if hasattr(result, 'pred_instances'):
                # MMDet v3.x format
                instances = result.pred_instances
                scores = instances.scores.cpu().numpy()
                labels = instances.labels.cpu().numpy()
                bboxes = instances.bboxes.cpu().numpy()
                
                for score, label, bbox in zip(scores, labels, bboxes):
                    if score > conf_threshold:
                        detections.append({
                            'label': int(label),
                            'score': float(score),
                            'bbox': bbox.tolist()
                        })
            else:
                # MMDet v2.x format
                if isinstance(result, tuple):
                    bbox_result, segm_result = result
                else:
                    bbox_result = result
                
                for class_id, class_detections in enumerate(bbox_result):
                    for detection in class_detections:
                        if len(detection) >= 5 and detection[4] > conf_threshold:
                            detections.append({
                                'label': class_id,
                                'score': float(detection[4]),
                                'bbox': detection[:4].tolist()
                            })
            
            return detections
        
        except Exception as e:
            print(f"Error detecting objects in {image_path}: {e}")
            return []
    
    def evaluate_object_presence(self, image_path, required_objects, class_names):
        """
        Evaluate if required objects are present in the generated image
        """
        detections = self.detect_objects(image_path)
        
        detected_classes = set()
        for detection in detections:
            class_name = class_names[detection['label']] if detection['label'] < len(class_names) else f"class_{detection['label']}"
            detected_classes.add(class_name.lower())
        
        # Check if required objects are present
        present_objects = []
        missing_objects = []
        
        for obj in required_objects:
            obj_lower = obj.lower()
            if obj_lower in detected_classes:
                present_objects.append(obj)
            else:
                missing_objects.append(obj)
        
        accuracy = len(present_objects) / len(required_objects) if required_objects else 0.0
        
        return {
            'accuracy': accuracy,
            'present_objects': present_objects,
            'missing_objects': missing_objects,
            'total_detected': len(detected_classes),
            'detections': detections
        }
    
    def evaluate_object_counting(self, image_path, object_counts, class_names):
        """
        Evaluate object counting accuracy
        """
        detections = self.detect_objects(image_path)
        
        # Count detected objects by class
        detected_counts = {}
        for detection in detections:
            class_name = class_names[detection['label']] if detection['label'] < len(class_names) else f"class_{detection['label']}"
            class_name = class_name.lower()
            detected_counts[class_name] = detected_counts.get(class_name, 0) + 1
        
        # Evaluate counting accuracy
        count_accuracy = []
        for obj, expected_count in object_counts.items():
            detected_count = detected_counts.get(obj.lower(), 0)
            is_correct = detected_count == expected_count
            count_accuracy.append(is_correct)
        
        overall_accuracy = sum(count_accuracy) / len(count_accuracy) if count_accuracy else 0.0
        
        return {
            'count_accuracy': overall_accuracy,
            'detected_counts': detected_counts,
            'expected_counts': object_counts,
            'correct_counts': sum(count_accuracy)
        }
    
    def evaluate_spatial_relationships(self, image_path, spatial_relations, class_names):
        """
        Evaluate spatial relationships between objects
        """
        detections = self.detect_objects(image_path)
        
        # Group detections by class
        objects_by_class = {}
        for detection in detections:
            class_name = class_names[detection['label']] if detection['label'] < len(class_names) else f"class_{detection['label']}"
            class_name = class_name.lower()
            if class_name not in objects_by_class:
                objects_by_class[class_name] = []
            objects_by_class[class_name].append(detection['bbox'])
        
        # Evaluate spatial relationships
        correct_relations = 0
        total_relations = len(spatial_relations)
        
        for relation in spatial_relations:
            obj1, relationship, obj2 = relation.get('obj1', ''), relation.get('relation', ''), relation.get('obj2', '')
            
            obj1_boxes = objects_by_class.get(obj1.lower(), [])
            obj2_boxes = objects_by_class.get(obj2.lower(), [])
            
            if obj1_boxes and obj2_boxes:
                # Check spatial relationship
                relation_satisfied = self._check_spatial_relationship(
                    obj1_boxes[0], obj2_boxes[0], relationship
                )
                if relation_satisfied:
                    correct_relations += 1
        
        spatial_accuracy = correct_relations / total_relations if total_relations > 0 else 0.0
        
        return {
            'spatial_accuracy': spatial_accuracy,
            'correct_relations': correct_relations,
            'total_relations': total_relations
        }
    
    def _check_spatial_relationship(self, bbox1, bbox2, relationship):
        """
        Check if two bounding boxes satisfy a spatial relationship
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate centers
        center1_x, center1_y = (x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2
        center2_x, center2_y = (x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2
        
        relationship = relationship.lower()
        
        if relationship in ['left of', 'to the left of']:
            return center1_x < center2_x
        elif relationship in ['right of', 'to the right of']:
            return center1_x > center2_x
        elif relationship in ['above', 'on top of']:
            return center1_y < center2_y
        elif relationship in ['below', 'under', 'underneath']:
            return center1_y > center2_y
        elif relationship in ['next to', 'beside']:
            horizontal_distance = abs(center1_x - center2_x)
            vertical_distance = abs(center1_y - center2_y)
            return horizontal_distance < vertical_distance * 2  # Heuristic
        else:
            return False  # Unknown relationship


def load_geneval_prompts(prompt_file):
    """
    Load GenEval prompts in JSONL format
    Each line contains: {"tag": "category", "include": [{"class": "object", "count": N}], "prompt": "text"}
    """
    prompts = []
    
    with open(prompt_file, 'r') as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            try:
                item = json.loads(line)
                
                # Extract objects and counts from "include" field
                objects = []
                object_counts = {}
                
                if 'include' in item:
                    for obj_info in item['include']:
                        obj_class = obj_info.get('class', '')
                        obj_count = obj_info.get('count', 1)
                        
                        objects.append(obj_class)
                        object_counts[obj_class] = obj_count
                
                prompt_info = {
                    'id': f"prompt_{line_idx:06d}",
                    'tag': item.get('tag', ''),
                    'prompt': item.get('prompt', ''),
                    'objects': objects,
                    'object_counts': object_counts,
                    'spatial_relations': [],  # Will be populated based on tag
                    'attributes': {}
                }
                
                # Parse spatial relationships from prompt text for certain tags
                if prompt_info['tag'] in ['position', 'spatial']:
                    prompt_info['spatial_relations'] = parse_spatial_relations_from_prompt(
                        prompt_info['prompt'], objects
                    )
                
                prompts.append(prompt_info)
                
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_idx + 1}: {e}")
                continue
    
    return prompts


def parse_spatial_relations_from_prompt(prompt_text, objects):
    """
    Parse spatial relationships from prompt text
    """
    spatial_relations = []
    
    # Common spatial relationship patterns
    spatial_keywords = {
        'left': ['left of', 'to the left of'],
        'right': ['right of', 'to the right of'], 
        'above': ['above', 'over', 'on top of'],
        'below': ['below', 'under', 'underneath'],
        'beside': ['next to', 'beside', 'near']
    }
    
    prompt_lower = prompt_text.lower()
    
    # Look for spatial relationships in prompt
    for direction, patterns in spatial_keywords.items():
        for pattern in patterns:
            if pattern in prompt_lower:
                # Try to extract objects involved
                if len(objects) >= 2:
                    spatial_relations.append({
                        'obj1': objects[0],
                        'relation': pattern,
                        'obj2': objects[1]
                    })
                break
    
    return spatial_relations


def generate_images_vila_u(model, prompts, output_dir, cfg_scale=3.0, generation_nums=1):
    """
    Generate images using VILA-U model for GenEval prompts
    Creates folder structure: <output_dir>/<prompt_idx>/samples/<sample_idx>.png
    """
    print(f"Generating images with VILA-U model...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    for prompt_idx, prompt_info in enumerate(tqdm(prompts, desc="Generating images")):
        prompt_text = prompt_info['prompt']
        
        # Create folder structure: 00000/, 00001/, etc.
        prompt_folder = os.path.join(output_dir, f"{prompt_idx:05d}")
        samples_folder = os.path.join(prompt_folder, "samples")
        os.makedirs(samples_folder, exist_ok=True)
        
        # Create metadata.jsonl with the current prompt info
        metadata_path = os.path.join(prompt_folder, "metadata.jsonl")
        with open(metadata_path, 'w') as f:
            # Write the original JSONL line
            original_line = {
                "tag": prompt_info['tag'],
                "include": [],
                "prompt": prompt_info['prompt']
            }
            # Reconstruct include field from objects and counts
            for obj in prompt_info['objects']:
                count = prompt_info['object_counts'].get(obj, 1)
                original_line["include"].append({"class": obj, "count": count})
            
            json.dump(original_line, f)
            f.write('\n')
        
        try:
            # Generate images using VILA-U
            response = model.generate_image_content(prompt_text, cfg_scale, generation_nums)
            
            # Save generated images in samples folder
            for i in range(response.shape[0]):
                image = response[i].permute(1, 2, 0)
                image = image.cpu().numpy().astype(np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Save as 0000.png, 0001.png, etc.
                sample_path = os.path.join(samples_folder, f"{i:04d}.png")
                cv2.imwrite(sample_path, image)
        
        except Exception as e:
            print(f"Error generating image for prompt {prompt_idx}: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(description='Evaluate VILA-U model using GenEval benchmark')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to VILA-U model')
    parser.add_argument('--img_path', type=str,
                        help='Path to directory containing generated images (for evaluation only)')
    parser.add_argument('--prompt_file', type=str, required=True,
                        help='Path to GenEval prompt JSONL file')
    parser.add_argument('--detector_model_path', type=str, required=True,
                        help='Path to MMDetection model weights')
    parser.add_argument('--detector_config_path', type=str, required=True,
                        help='Path to MMDetection config file')
    parser.add_argument('--class_names_file', type=str, 
                        help='Path to file containing class names')
    parser.add_argument('--conf_threshold', type=float, default=0.3,
                        help='Confidence threshold for object detection')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for computation')
    parser.add_argument('--exp_name', type=str, default='geneval_experiment',
                        help='Experiment name')
    parser.add_argument('--report_to', type=str, default='wandb',
                        help='Where to report results')
    parser.add_argument('--name', type=str, default='vila_u_geneval',
                        help='Run name for tracking')
    parser.add_argument('--tracker_project_name', type=str, default='vila-u-evaluation',
                        help='Project name for tracking')
    parser.add_argument('--log_geneval', action='store_true',
                        help='Log GenEval results to tracker')
    parser.add_argument('--output_dir', type=str, default='./geneval_vila_u_results',
                        help='Output directory for generated images and results')
    parser.add_argument('--cfg_scale', type=float, default=3.0,
                        help='CFG scale for VILA-U image generation')
    parser.add_argument('--generation_nums', type=int, default=1,
                        help='Number of images to generate per prompt')
    parser.add_argument('--generate_only', action='store_true',
                        help='Only generate images, skip evaluation')
    parser.add_argument('--evaluate_only', action='store_true',
                        help='Only evaluate existing images, skip generation')

    args = parser.parse_args()

    if not MMDET_AVAILABLE and not args.generate_only:
        print("MMDetection is not available. Please install it for object detection evaluation.")
        return

    # Load class names
    if args.class_names_file and os.path.exists(args.class_names_file):
        with open(args.class_names_file, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
    else:
        # Default COCO class names
        class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light']
    
    # Load prompts
    print(f"Loading GenEval prompts from {args.prompt_file}")
    prompts = load_geneval_prompts(args.prompt_file)
    print(f"Loaded {len(prompts)} prompts")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load VILA-U model if needed
    if not args.evaluate_only:
        print(f"Loading VILA-U model from {args.model_path}")
        model = vila_u.load(args.model_path)
        
        # Generate images
        image_output_dir = os.path.join(args.output_dir, 'generated_images')
        generate_images_vila_u(model, prompts, image_output_dir, args.cfg_scale, args.generation_nums)
        
        if args.generate_only:
            print(f"Image generation completed. Images saved to {image_output_dir}")
            return
    else:
        if not args.img_path:
            raise ValueError("Must specify --img_path when using --evaluate_only")
        image_output_dir = args.img_path
    
    # Initialize evaluator
    evaluator = GenEvalEvaluator(
        model_path=args.detector_model_path,
        config_path=args.detector_config_path,
        device=args.device
    )
    
    # Evaluate each prompt
    results = {
        'tag_statistics': {},
        'overall_accuracy': 0.0,
        'total_samples': 0,
        'total_correct': 0
    }
    
    # Track results by tag category
    tag_results = {}
    
    for prompt_idx, prompt_info in enumerate(tqdm(prompts, desc="Evaluating GenEval prompts")):
        tag = prompt_info['tag']
        
        # Look for generated images in new folder structure
        prompt_folder = os.path.join(image_output_dir, f"{prompt_idx:05d}")
        samples_folder = os.path.join(prompt_folder, "samples")
        
        # Find the first available image in samples folder
        image_path = None
        if os.path.exists(samples_folder):
            for i in range(args.generation_nums):
                potential_path = os.path.join(samples_folder, f"{i:04d}.png")
                if os.path.exists(potential_path):
                    image_path = potential_path
                    break
        
        if not image_path:
            print(f"Warning: Image not found for prompt {prompt_idx}: {samples_folder}")
            continue
        
        # Initialize tag tracking
        if tag not in tag_results:
            tag_results[tag] = {'correct': 0, 'total': 0}
        
        tag_results[tag]['total'] += 1
        results['total_samples'] += 1
        
        # Evaluate based on tag type
        success = False
        
        if tag == 'single_object':
            # For single object, check if the object is present
            if prompt_info['objects']:
                presence_result = evaluator.evaluate_object_presence(
                    image_path, prompt_info['objects'], class_names
                )
                success = presence_result['accuracy'] >= 0.5  # At least 50% of objects found
        
        elif tag == 'two_object':
            # For two objects, check if both are present
            if prompt_info['objects']:
                presence_result = evaluator.evaluate_object_presence(
                    image_path, prompt_info['objects'], class_names
                )
                success = presence_result['accuracy'] >= 0.9  # Almost all objects found
        
        elif tag == 'counting':
            # For counting, check exact count accuracy
            if prompt_info['object_counts']:
                counting_result = evaluator.evaluate_object_counting(
                    image_path, prompt_info['object_counts'], class_names
                )
                success = counting_result['count_accuracy'] >= 0.8  # High accuracy for counting
        
        elif tag == 'colors':
            # For colors, check object presence (color detection would need additional model)
            if prompt_info['objects']:
                presence_result = evaluator.evaluate_object_presence(
                    image_path, prompt_info['objects'], class_names
                )
                success = presence_result['accuracy'] >= 0.7
        
        elif tag == 'position':
            # For position, check spatial relationships
            if prompt_info['spatial_relations']:
                spatial_result = evaluator.evaluate_spatial_relationships(
                    image_path, prompt_info['spatial_relations'], class_names
                )
                success = spatial_result['spatial_accuracy'] >= 0.6
            elif prompt_info['objects']:
                # Fallback to object presence if no spatial relations parsed
                presence_result = evaluator.evaluate_object_presence(
                    image_path, prompt_info['objects'], class_names
                )
                success = presence_result['accuracy'] >= 0.7
        
        elif tag == 'color_attr':
            # For color attributes, check object presence
            if prompt_info['objects']:
                presence_result = evaluator.evaluate_object_presence(
                    image_path, prompt_info['objects'], class_names
                )
                success = presence_result['accuracy'] >= 0.7
        
        else:
            # Default: check object presence
            if prompt_info['objects']:
                presence_result = evaluator.evaluate_object_presence(
                    image_path, prompt_info['objects'], class_names
                )
                success = presence_result['accuracy'] >= 0.5
        
        # Update statistics
        if success:
            tag_results[tag]['correct'] += 1
            results['total_correct'] += 1
    
    # Calculate tag-specific accuracies
    for tag, stats in tag_results.items():
        accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
        results[f'{tag}_accuracy'] = accuracy
        results['tag_statistics'][tag] = stats
    
    # Calculate overall accuracy
    overall_accuracy = results['total_correct'] / results['total_samples'] if results['total_samples'] > 0 else 0.0
    results['overall_accuracy'] = overall_accuracy
    
    # Print results
    print("\nGenEval Results:")
    print(f"Overall Accuracy: {overall_accuracy:.4f} ({results['total_correct']}/{results['total_samples']})")
    print("\nTag-specific Results:")
    for tag, stats in tag_results.items():
        accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
        print(f"  {tag}: {accuracy:.4f} ({stats['correct']}/{stats['total']})")
    
    # Save results
    output_file = os.path.join(args.output_dir, f"{args.exp_name}_geneval_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    
    # Log to tracker
    if args.log_geneval and args.report_to == 'wandb':
        result_dict = {args.exp_name: overall_accuracy}
        tracker(args, result_dict, label="", pattern="epoch_step", metric="GenEval")


if __name__ == '__main__':
    main()