import numpy as np
import cv2
import os
from datetime import datetime

class ThermalSatelliteGenerator:
    def __init__(self, width=640, height=640, base_temp_celsius=45):
        """
        Initialize thermal image generator for 4U satellite ceiling
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            base_temp_celsius: Base temperature of ceiling (40-50°C range)
        """
        self.width = width
        self.height = height
        self.base_temp = base_temp_celsius
        self.temp_range = 10  # ±5°C from base temperature
        
    def create_base_temperature_map(self):
        """Create realistic base temperature distribution for satellite ceiling"""
        # Start with base temperature
        temp_map = np.ones((self.height, self.width), dtype=np.float32) * self.base_temp
        
        # Add thermal stratification (heat rises - top slightly warmer)
        vertical_gradient = np.linspace(-1, 1, self.height).reshape(-1, 1)
        temp_map += vertical_gradient * 1.5  # 3°C difference top to bottom
        
        return temp_map
    
    def add_aluminum_panel_patterns(self, temp_map):
        """Add realistic thermal patterns for aluminum satellite panels"""
        # Create coordinate grids
        Y, X = np.ogrid[:self.height, :self.width]
        
        # Structural support beams (thermal bridging)
        # Create grid pattern of slightly cooler lines where structure conducts heat away
        beam_spacing = 120
        
        # Vertical beams
        for i in range(0, self.width, beam_spacing):
            beam_mask = np.abs(X - i) < 3
            temp_map[beam_mask.squeeze()] -= 0.8  # Beams are slightly cooler
        
        # Horizontal beams  
        for i in range(0, self.height, beam_spacing):
            beam_mask = np.abs(Y - i) < 3
            temp_map[beam_mask.squeeze()] -= 0.8
        
        # Add panel joints with slight temperature variation
        panel_size = 200
        
        # Vertical joints
        for i in range(panel_size, self.width, panel_size):
            joint_mask = np.abs(X - i) < 2
            temp_map[joint_mask.squeeze()] -= 0.5
            
        # Horizontal joints
        for i in range(panel_size, self.height, panel_size):
            joint_mask = np.abs(Y - i) < 2
            temp_map[joint_mask.squeeze()] -= 0.5
            
        return temp_map
        
    def add_electronic_hotspots(self, temp_map):
        """Add heat patterns from internal electronics"""
        np.random.seed(None)  # Random each time
        
        # Major heat sources (processors, power supplies)
        num_major_sources = np.random.randint(2, 4)
        for _ in range(num_major_sources):
            cx = np.random.randint(self.width//4, 3*self.width//4)
            cy = np.random.randint(self.height//4, 3*self.height//4)
            
            # Heat spreads in aluminum with characteristic pattern
            Y, X = np.ogrid[:self.height, :self.width]
            dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
            
            # Heat source with thermal spreading
            heat_intensity = np.random.uniform(1, 2)  # 3-5°C above base
            heat_size = np.random.uniform(20, 90)
            heat_pattern = heat_intensity * np.exp(-(dist**2) / (2 * heat_size**2))
            
            # Aluminum spreads heat, so add wider gradual spread
            spread_pattern = (heat_intensity * 0.3) * np.exp(-(dist**2) / (2 * (heat_size * 3)**2))
            
            temp_map += heat_pattern + spread_pattern
        
        # Minor heat sources (smaller components)
        num_minor_sources = np.random.randint(5, 8)
        for _ in range(num_minor_sources):
            cx = np.random.randint(50, self.width-50)
            cy = np.random.randint(50, self.height-50)
            
            Y, X = np.ogrid[:self.height, :self.width]
            dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
            
            heat_intensity = np.random.uniform(1, 2.5)
            heat_size = np.random.uniform(20, 35)
            temp_map += heat_intensity * np.exp(-(dist**2) / (2 * heat_size**2))
        
        return temp_map
    
    def add_ventilation_patterns(self, temp_map):
        """Add cooling patterns from ventilation"""
        # Cooling vents create localized cool spots
        num_vents = np.random.randint(1, 2)  # Only 1-2 vents
        
        for _ in range(num_vents):
            vent_x = np.random.randint(100, self.width-100)
            vent_y = np.random.randint(100, self.height-100)
            
            Y, X = np.ogrid[:self.height, :self.width]
            
            # MUCH smaller circular vent pattern
            dist = np.sqrt((X - vent_x)**2 + (Y - vent_y)**2)
            vent_cooling = -0.5 * np.exp(-(dist**2) / (2 * 4**2))  # Tiny 4 pixel radius, reduced intensity
            
            # Minimal air flow pattern
            flow_angle = np.random.uniform(0, 2*np.pi)
            flow_x = np.cos(flow_angle)
            flow_y = np.sin(flow_angle)
            
            # Very subtle flow pattern
            flow_dist = (X - vent_x) * flow_x + (Y - vent_y) * flow_y
            flow_pattern = -0.1 * np.exp(-(flow_dist**2) / (2 * 30**2)) * np.exp(-(dist**2) / (2 * 15**2))
            
            temp_map += vent_cooling + flow_pattern
        
        return temp_map
    
    def add_material_variations(self, temp_map):
        """Add subtle variations from different materials and emissivities"""
        # Different surface treatments have different emissivities
        # Create patches with slightly different apparent temperatures
        
        num_patches = np.random.randint(3, 6)
        for _ in range(num_patches):
            patch_x = np.random.randint(0, self.width-150)
            patch_y = np.random.randint(0, self.height-150)
            patch_w = np.random.randint(50, 150)
            patch_h = np.random.randint(50, 150)
            
            # Different material appears slightly different temperature
            emissivity_variation = np.random.uniform(-0.5, 0.5)
            temp_map[patch_y:patch_y+patch_h, patch_x:patch_x+patch_w] += emissivity_variation
            
            # Smooth edges
            temp_map = cv2.GaussianBlur(temp_map, (15, 15), 5)
        
        return temp_map
    
    def add_realistic_noise(self, temp_map):
        """Add thermal camera noise characteristics"""
        # NETD noise (40-50mK for good cameras)
        netd = 0.045  # 45mK in Celsius
        temporal_noise = np.random.normal(0, netd, temp_map.shape)
        
        # Fixed pattern noise (pixel-to-pixel sensitivity variation)
        fixed_pattern = np.random.normal(1, 0.02, temp_map.shape)  # ±2% variation
        
        # Apply noise
        temp_map = temp_map * fixed_pattern + temporal_noise
        
        # Add very subtle 1/f noise for realism
        low_freq_noise = cv2.resize(
            np.random.randn(self.height//10, self.width//10) * 0.1,
            (self.width, self.height),
            interpolation=cv2.INTER_CUBIC
        )
        temp_map += low_freq_noise
        
        return temp_map
    
    def apply_thermal_colormap(self, temp_map):
        """Apply realistic FLIR Ironbow colormap"""
        # Normalize temperature to 0-255 for colormap
        temp_min = self.base_temp - self.temp_range/2
        temp_max = self.base_temp + self.temp_range/2
        
        # Clip and normalize
        temp_map = np.clip(temp_map, temp_min, temp_max)
        normalized = ((temp_map - temp_min) / (temp_max - temp_min) * 255).astype(np.uint8)
        
        # Create proper Ironbow colormap (dark to bright)
        colormap = np.zeros((256, 3), dtype=np.uint8)
        
        # Ironbow: black -> purple -> red -> orange -> yellow -> white
        # In apply_thermal_colormap method, update the color points:
        points = [
            (0, (60, 50, 70)),        # Darker purple (coldest - no black!)
            (40, (70, 40, 80)),       # Dark purple
            (80, (90, 30, 85)),       # Purple
            (100, (120, 20, 60)),     # Red-purple
            (120, (160, 20, 20)),     # Dark red
            (140, (200, 40, 20)),     # Red
            (160, (220, 80, 20)),     # Red-orange
            (180, (240, 120, 20)),    # Orange
            (200, (250, 180, 40)),    # Yellow-orange
            (220, (255, 220, 80)),    # Yellow
            (240, (255, 240, 160)),   # Pale yellow
            (255, (255, 255, 220))    # White-yellow (hottest)
        ]
        # Interpolate between points
        for i in range(len(points) - 1):
            start_idx, start_color = points[i]
            end_idx, end_color = points[i + 1]
            
            for j in range(start_idx, end_idx):
                t = (j - start_idx) / (end_idx - start_idx)
                color = [
                    int((1 - t) * start_color[k] + t * end_color[k])
                    for k in range(3)
                ]
                colormap[j] = color
        
        # Apply colormap
        thermal_image = colormap[normalized]
        
        # Convert RGB to BGR for OpenCV
        thermal_image = cv2.cvtColor(thermal_image, cv2.COLOR_RGB2BGR)
        
        return thermal_image, temp_map
    
    def add_camera_effects(self, thermal_image):
        """Add final camera-specific effects"""
        # Slight vignetting (darker at edges)
        Y, X = np.ogrid[:self.height, :self.width]
        center_x, center_y = self.width/2, self.height/2
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        vignette = 1 - (dist_from_center / max_dist) * 0.15  # 15% darkening at corners
        
        # Apply vignetting
        thermal_image = (thermal_image * vignette[:, :, np.newaxis]).astype(np.uint8)
        
        # Very slight motion blur (camera not perfectly still)
        kernel_size = 3
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
        thermal_image = cv2.filter2D(thermal_image, -1, kernel)
        
        # Remove the UI elements - don't call add_thermal_ui anymore
        # if np.random.random() > 0.5:  # 50% chance to add UI elements
        #     self.add_thermal_ui(thermal_image)
        
        return thermal_image
    
    def add_thermal_ui(self, image):
        """Add realistic thermal camera UI elements"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Temperature range indicator
        temp_text = f"{self.base_temp-self.temp_range/2:.1f}C - {self.base_temp+self.temp_range/2:.1f}C"
        cv2.putText(image, temp_text, (10, 25), font, 0.6, (255, 255, 255), 1)
        
        # Emissivity setting (typical for metal)
        cv2.putText(image, "e=0.85", (10, self.height-10), font, 0.5, (255, 255, 255), 1)
        
        # Add crosshair at center
        cx, cy = self.width // 2, self.height // 2
        cv2.line(image, (cx - 20, cy), (cx - 5, cy), (255, 255, 255), 1)
        cv2.line(image, (cx + 5, cy), (cx + 20, cy), (255, 255, 255), 1)
        cv2.line(image, (cx, cy - 20), (cx, cy - 5), (255, 255, 255), 1)
        cv2.line(image, (cx, cy + 5), (cx, cy + 20), (255, 255, 255), 1)
        
        # Center point temperature
        cv2.putText(image, f"{self.base_temp:.1f}C", (cx + 25, cy - 10), 
                   font, 0.5, (255, 255, 255), 1)
        
        return image
    
    def generate(self):
        """Generate a complete thermal image"""
        # Create base temperature map
        temp_map = self.create_base_temperature_map()
        
        # Add various thermal patterns
        temp_map = self.add_aluminum_panel_patterns(temp_map)
        temp_map = self.add_electronic_hotspots(temp_map)
        temp_map = self.add_ventilation_patterns(temp_map)
        temp_map = self.add_material_variations(temp_map)
        
        # Add realistic noise
        temp_map = self.add_realistic_noise(temp_map)
        
        # Apply thermal colormap
        thermal_image, final_temp_map = self.apply_thermal_colormap(temp_map)
        
        # Add camera effects
        thermal_image = self.add_camera_effects(thermal_image)
        
        return thermal_image, final_temp_map

def generate_thermal_dataset(num_images, file_name):
    """Generate multiple thermal images with variations"""
    output_dir = file_name
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating {num_images} thermal images...")
    
    for i in range(num_images):
        # Vary base temperature within 40-50°C range
        base_temp = np.random.uniform(40, 50)
        
        # Create generator with random parameters
        generator = ThermalSatelliteGenerator(
            width=640,
            height=640,
            base_temp_celsius=base_temp
        )
        
        # Generate image
        thermal_image, temp_map = generator.generate()
        
        # Save image
        filename = f'thermal_satellite_{i:04d}.png'
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, thermal_image)

        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{num_images} images...")
    
    print(f"Dataset saved to {output_dir}/")

def main_create_thermal_ceiling(num_images, file_name):
    generate_thermal_dataset(num_images, file_name)