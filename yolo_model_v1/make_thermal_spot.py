import cv2
import numpy as np
import os
import random

class ThermalSimulator:
    def __init__(self, image_path):
        self.original = cv2.imread(image_path)
        self.thermal_map = self.original.copy()
        self.cold_spots = []
        self.hot_spots = []
        
    def update_thermal(self):
        # Start with original image
        self.thermal_map = self.original.copy()
        
        # Convert to float for calculations
        temp_map = self.thermal_map.astype(float)
        
        # Apply cold spots with natural gradient
        for x, y, radius, intensity in self.cold_spots:
            # Create gradient mask
            Y, X = np.ogrid[:temp_map.shape[0], :temp_map.shape[1]]
            dist = np.sqrt((X - x)**2 + (Y - y)**2)
            
            # Multi-layer gradient for smooth transition
            # Outer layer - very subtle cooling
            outer_mask = np.exp(-(dist**2) / (2 * (radius * 2)**2))
            
            # Middle layer - moderate cooling
            mid_mask = np.exp(-(dist**2) / (2 * (radius * 1.2)**2))
            
            # Inner layer - cold core
            inner_mask = np.exp(-(dist**2) / (2 * (radius * 0.6)**2))
            
            # Very inner core
            core_mask = np.exp(-(dist**2) / (2 * (radius * 0.3)**2))

        # Outer: light blue tint
        temp_map[:, :, 0] = temp_map[:, :, 0] + (outer_mask * intensity * 0.85)  # Blue increase
        temp_map[:, :, 1] = temp_map[:, :, 1] + (outer_mask * intensity * 0.05)  # Slight green increase
        temp_map[:, :, 2] = temp_map[:, :, 2] - (outer_mask * intensity * 0.05)  # Slight red reduction

        # Middle: more pronounced light blue
        temp_map[:, :, 0] = temp_map[:, :, 0] + (mid_mask * intensity * 0.55)    # More blue
        temp_map[:, :, 1] = temp_map[:, :, 1] + (mid_mask * intensity * 0.1)     # Keep some green
        temp_map[:, :, 2] = temp_map[:, :, 2] - (mid_mask * intensity * 0.1)     # Reduce red

        # Inner: bright light blue
        temp_map[:, :, 0] = temp_map[:, :, 0] + (inner_mask * intensity * 0.85)  # Strong blue
        temp_map[:, :, 1] = temp_map[:, :, 1] + (inner_mask * intensity * 0.15)  # Some green for lightness
        temp_map[:, :, 2] = temp_map[:, :, 2] - (inner_mask * intensity * 0.15)  # Reduce red

        # Core: brightest light blue center
        temp_map[:, :, 0] = temp_map[:, :, 0] + (core_mask * intensity * 0.2)    # Peak blue
        temp_map[:, :, 1] = temp_map[:, :, 1] + (core_mask * intensity * 0.2)    # Keep green for brightness
        temp_map[:, :, 2] = temp_map[:, :, 2] - (core_mask * intensity * 0.2)  
        
        # Clip values
        self.thermal_map = np.clip(temp_map, 0, 255).astype(np.uint8)
        
        # Apply slight blur to make transitions even smoother
        self.thermal_map = cv2.GaussianBlur(self.thermal_map, (5, 5), 1)
    
    def add_cold_spot(self, x, y, radius=50, intensity=100):
        self.cold_spots.append((x, y, radius, intensity))
        self.update_thermal()

def add_spots_to_all_images(input_dir, output_dir):
    """Add subtle gradient cold spots to all thermal images"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all thermal images
    image_files = [f for f in os.listdir(input_dir) if f.startswith('thermal_satellite_') and f.endswith('.png')]
    image_files.sort()
    
    print(f"Found {len(image_files)} images to process...")
    
    for i, filename in enumerate(image_files):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        # Create simulator for this image
        sim = ThermalSimulator(input_path)
        
        # Add 1-2 cold spots (fewer for subtlety)
        num_spots = 1
        
        for _ in range(num_spots):
            # Random position (avoiding edges)
            x = random.randint(120, 520)  # For 640x640 images
            y = random.randint(120, 520)
            
            # Moderate size and intensity for subtle effect
            radius = 45
            # intensity = random.randint(80, 120)  # Moderate intensity
            intensity = 130
            
            # Add cold spot with gradient
            sim.add_cold_spot(x, y, radius, intensity)
        
        # Save the modified image
        cv2.imwrite(output_path, sim.thermal_map)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(image_files)} images...")
    
    print(f"All images processed! Output saved to '{output_dir}' folder")

# Run the batch processing
def main_make_thermal_spot(ipt_dir, opt_dir):
    add_spots_to_all_images(ipt_dir, opt_dir)