"""
Pipeline de Classificação de Grãos de MILHO usando OpenCV
=========================================================

Este script implementa uma pipeline completa de processamento de imagem
especificamente otimizada para segmentar, extrair características e 
classificar grãos de milho.

Características específicas do milho consideradas:
- Cor: amarelo intenso a laranja (diferente da soja que é bege)
- Forma: mais achatado e angular que a soja
- Defeitos comuns: ardidos, mofados, fermentados, carunchados, quebrados

Classificação baseada na IN 60/2011 MAPA (simplificada):
- Tipo 1: Máximo 1% defeitos
- Tipo 2: Máximo 2% defeitos  
- Tipo 3: Máximo 3% defeitos
- Fora de Tipo: Acima de 3% defeitos

Autor: Pipeline gerada com Claude
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from enum import Enum
import json
import os


class CornQuality(Enum):
    """Classificação de qualidade dos grãos de milho"""
    TIPO_1 = "Tipo 1 (Premium)"
    TIPO_2 = "Tipo 2 (Bom)"
    TIPO_3 = "Tipo 3 (Regular)"
    FORA_TIPO = "Fora de Tipo"
    DESCARTE = "Descarte"


class CornDefect(Enum):
    """Tipos de defeitos em grãos de milho (baseado em normas brasileiras)"""
    NENHUM = "Nenhum"
    ARDIDO = "Ardido"              # Fermentação, cor escurecida
    MOFADO = "Mofado"              # Presença de fungos
    FERMENTADO = "Fermentado"      # Alteração de cor por fermentação
    GERMINADO = "Germinado"        # Grão que começou a germinar
    CARUNCHADO = "Carunchado"      # Atacado por insetos
    CHOCO = "Choco"                # Imaturo, enrugado
    QUEBRADO = "Quebrado"          # Fragmentado
    DESCOLORIDO = "Descolorido"    # Perda da cor característica
    MANCHADO = "Manchado"          # Manchas escuras


@dataclass
class CornGrainFeatures:
    """Características extraídas de cada grão de milho"""
    # Identificação
    id: int
    
    # Geometria
    area: float
    perimeter: float
    circularity: float
    aspect_ratio: float
    solidity: float
    extent: float
    equivalent_diameter: float
    
    # Dimensões
    width: float
    height: float
    major_axis: float
    minor_axis: float
    
    # Cor (espaço HSV) - crucial para milho
    mean_hue: float
    mean_saturation: float
    mean_value: float
    std_hue: float
    std_saturation: float
    std_value: float
    
    # Cor (espaço LAB)
    mean_l: float
    mean_a: float
    mean_b: float
    
    # Cor adicional (RGB para análise de amarelo)
    mean_r: float
    mean_g: float
    mean_b_rgb: float
    yellow_index: float  # Índice de amarelecimento
    
    # Textura
    texture_contrast: float
    texture_homogeneity: float
    texture_energy: float
    
    # Análise de manchas
    dark_spot_ratio: float      # Proporção de pixels escuros
    color_uniformity: float     # Uniformidade de cor
    
    # Posição
    centroid: Tuple[int, int]
    bounding_box: Tuple[int, int, int, int]
    
    # Classificação
    quality: CornQuality = CornQuality.TIPO_2
    defects: List[CornDefect] = field(default_factory=list)
    confidence: float = 0.0
    defect_score: float = 0.0


class CornClassifierPipeline:
    """
    Pipeline completa para classificação de grãos de MILHO usando processamento de imagem.
    
    Otimizada para:
    - Detecção de cor amarela característica do milho
    - Identificação de defeitos específicos (ardidos, mofados, etc.)
    - Classificação conforme padrões brasileiros
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Inicializa a pipeline com configurações para milho.
        
        Args:
            config: Dicionário com parâmetros de configuração
        """
        self.config = config or self._default_config()
        self.grains: List[CornGrainFeatures] = []
        self.original_image = None
        self.processed_image = None
        self.mask = None
        
    def _default_config(self) -> Dict:
        """Retorna configurações padrão otimizadas para MILHO"""
        return {
            # Pré-processamento
            "blur_kernel_size": 5,
            "bilateral_d": 9,
            "bilateral_sigma_color": 75,
            "bilateral_sigma_space": 75,
            
            # Segmentação - ajustado para cor do milho
            "threshold_method": "corn_color",
            "morph_kernel_size": 3,
            "min_grain_area": 150,
            "max_grain_area": 150000,
            
            # Range de cor HSV para milho amarelo
            # Milho tem matiz entre 15-35 (amarelo-laranja)
            "corn_hue_min": 15,
            "corn_hue_max": 40,
            "corn_sat_min": 80,
            "corn_sat_max": 255,
            "corn_val_min": 100,
            "corn_val_max": 255,
            
            # Watershed
            "use_watershed": True,
            "dist_transform_mask": 5,
            "watershed_threshold": 0.35,
            
            # Hough Circles para grãos próximos
            "hough_dp": 1.2,
            "hough_min_dist": 15,
            "hough_param1": 50,
            "hough_param2": 25,
            "hough_min_radius": 8,
            "hough_max_radius": 40,
            
            # Classificação - baseado em milho
            # Milho é mais achatado que soja
            "circularity_threshold_good": 0.65,
            "circularity_threshold_regular": 0.50,
            
            # Aspect ratio do milho (mais alongado)
            "aspect_ratio_min": 0.5,
            "aspect_ratio_max": 0.95,
            
            # Limiares para defeitos
            "dark_spot_threshold": 0.15,        # 15% de área escura = defeito
            "color_variation_threshold": 25,
            "ardido_value_threshold": 120,       # Grãos ardidos são mais escuros
            "yellow_index_min": 1.2,             # Índice mínimo de amarelo
            
            # Classificação por tipo (% máximo de defeitos)
            "tipo1_max_defects": 0.01,
            "tipo2_max_defects": 0.02,
            "tipo3_max_defects": 0.03,
        }
    
    # =========================================================================
    # ETAPA 1: PRÉ-PROCESSAMENTO
    # =========================================================================
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Aplica pré-processamento otimizado para grãos de milho.
        
        Técnicas aplicadas:
        - Redução de ruído com filtro bilateral
        - Realce de contraste com CLAHE
        - Realce do canal amarelo
        
        Args:
            image: Imagem BGR de entrada
            
        Returns:
            Imagem pré-processada
        """
        self.original_image = image.copy()
        
        # 1. Filtro bilateral - remove ruído preservando bordas
        filtered = cv2.bilateralFilter(
            image,
            d=self.config["bilateral_d"],
            sigmaColor=self.config["bilateral_sigma_color"],
            sigmaSpace=self.config["bilateral_sigma_space"]
        )
        
        # 2. Converter para LAB e aplicar CLAHE no canal L
        lab = cv2.cvtColor(filtered, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        # 3. Realçar o canal amarelo (importante para milho)
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Aumentar saturação levemente para destacar o amarelo
        s_enhanced = cv2.add(s, 10)
        s_enhanced = np.clip(s_enhanced, 0, 255).astype(np.uint8)
        
        hsv_enhanced = cv2.merge([h, s_enhanced, v])
        enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
        
        # 4. Suavização leve
        kernel_size = self.config["blur_kernel_size"]
        smoothed = cv2.GaussianBlur(enhanced, (kernel_size, kernel_size), 0)
        
        self.processed_image = smoothed
        return smoothed
    
    # =========================================================================
    # ETAPA 2: SEGMENTAÇÃO ESPECÍFICA PARA MILHO
    # =========================================================================
    
    def segment(self, image: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Segmenta a imagem para identificar grãos de milho.
        
        Usa detecção de cor amarela específica do milho.
        
        Args:
            image: Imagem pré-processada
            
        Returns:
            Máscara binária e lista de contornos
        """
        method = self.config["threshold_method"]
        
        if method == "corn_color":
            mask = self._segment_corn_color(image)
        elif method == "combined":
            mask = self._segment_combined(image)
        else:
            mask = self._segment_corn_color(image)
        
        # Aplicar operações morfológicas
        mask = self._morphological_cleanup(mask)
        
        # Usar watershed se configurado
        if self.config["use_watershed"]:
            mask = self._apply_watershed(image, mask)
        
        self.mask = mask
        
        # Encontrar contornos
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filtrar por área
        min_area = self.config["min_grain_area"]
        max_area = self.config["max_grain_area"]
        
        valid_contours = [
            cnt for cnt in contours
            if min_area < cv2.contourArea(cnt) < max_area
        ]
        
        # Se poucos contornos, tentar Hough Circles
        total_mask_area = np.sum(mask > 0)
        if len(valid_contours) <= 2 and total_mask_area > 5000:
            hough_contours = self._detect_grains_hough(image, mask)
            if len(hough_contours) > len(valid_contours):
                valid_contours = hough_contours
        
        return mask, valid_contours
    
    def _segment_corn_color(self, image: np.ndarray) -> np.ndarray:
        """
        Segmentação específica para cor do milho (amarelo/laranja).
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Range para milho amarelo
        lower_corn = np.array([
            self.config["corn_hue_min"],
            self.config["corn_sat_min"],
            self.config["corn_val_min"]
        ])
        upper_corn = np.array([
            self.config["corn_hue_max"],
            self.config["corn_sat_max"],
            self.config["corn_val_max"]
        ])
        
        mask_yellow = cv2.inRange(hsv, lower_corn, upper_corn)
        
        # Também detectar milho mais alaranjado
        lower_orange = np.array([5, 100, 100])
        upper_orange = np.array([20, 255, 255])
        mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
        
        # Combinar
        mask = cv2.bitwise_or(mask_yellow, mask_orange)
        
        return mask
    
    def _segment_combined(self, image: np.ndarray) -> np.ndarray:
        """
        Segmentação combinada para milho com fundo variável.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Método 1: Cor do milho
        mask_corn = self._segment_corn_color(image)
        
        # Método 2: Otsu invertido (para fundo claro)
        _, mask_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Método 3: Canal b do LAB (destaca amarelo)
        b_channel = lab[:, :, 2]
        _, mask_lab = cv2.threshold(b_channel, 140, 255, cv2.THRESH_BINARY)
        
        # Combinar com prioridade para detecção de cor do milho
        mask = cv2.bitwise_and(mask_corn, mask_otsu)
        mask = cv2.bitwise_or(mask, cv2.bitwise_and(mask_lab, mask_otsu))
        
        return mask
    
    def _morphological_cleanup(self, mask: np.ndarray) -> np.ndarray:
        """Aplica operações morfológicas para limpar a máscara"""
        kernel_size = self.config["morph_kernel_size"]
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        
        # Fechamento para preencher buracos
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Abertura para remover ruído
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return mask
    
    def _apply_watershed(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Aplica watershed para separar grãos conectados.
        """
        # Distance transform
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, self.config["dist_transform_mask"])
        
        # Normalizar
        dist_max = dist.max()
        if dist_max > 0:
            dist_normalized = dist / dist_max
        else:
            return mask
        
        # Threshold
        threshold = self.config["watershed_threshold"]
        _, sure_fg = cv2.threshold(dist_normalized, threshold, 1.0, cv2.THRESH_BINARY)
        sure_fg = np.uint8(sure_fg * 255)
        
        # Se muito poucos foreground, usar threshold menor
        if np.sum(sure_fg > 0) < 100:
            _, sure_fg = cv2.threshold(dist_normalized, threshold * 0.5, 1.0, cv2.THRESH_BINARY)
            sure_fg = np.uint8(sure_fg * 255)
        
        # Background
        kernel = np.ones((3, 3), np.uint8)
        sure_bg = cv2.dilate(mask, kernel, iterations=3)
        
        # Unknown region
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Markers
        num_labels, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # Watershed
        image_bgr = image if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(image_bgr, markers)
        
        # Final mask
        result_mask = np.zeros_like(mask)
        result_mask[markers > 1] = 255
        
        return result_mask
    
    def _detect_grains_hough(self, image: np.ndarray, mask: np.ndarray) -> List[np.ndarray]:
        """
        Detecta grãos usando Hough Circle Transform.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=self.config["hough_dp"],
            minDist=self.config["hough_min_dist"],
            param1=self.config["hough_param1"],
            param2=self.config["hough_param2"],
            minRadius=self.config["hough_min_radius"],
            maxRadius=self.config["hough_max_radius"]
        )
        
        contours = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                x, y, r = circle
                if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                    if mask[y, x] > 0:
                        # Criar contorno elíptico (milho é mais oval)
                        angles = np.linspace(0, 2 * np.pi, 36)
                        # Milho é ~20% mais alongado
                        rx = r
                        ry = int(r * 0.8)
                        pts = np.array([
                            [int(x + rx * np.cos(a)), int(y + ry * np.sin(a))]
                            for a in angles
                        ], dtype=np.int32)
                        contours.append(pts.reshape(-1, 1, 2))
        
        return contours
    
    # =========================================================================
    # ETAPA 3: EXTRAÇÃO DE CARACTERÍSTICAS PARA MILHO
    # =========================================================================
    
    def extract_features(
        self, image: np.ndarray, contours: List[np.ndarray]
    ) -> List[CornGrainFeatures]:
        """
        Extrai características específicas para grãos de milho.
        """
        self.grains = []
        
        # Converter para diferentes espaços de cor
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        for i, contour in enumerate(contours):
            # Criar máscara para o grão individual
            grain_mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(grain_mask, [contour], -1, 255, -1)
            
            # Extrair características
            geom_features = self._extract_geometric_features(contour)
            color_features = self._extract_color_features(image, hsv, lab, grain_mask)
            texture_features = self._extract_texture_features(gray, grain_mask)
            defect_features = self._extract_defect_features(image, hsv, gray, grain_mask)
            
            # Bounding box e centróide
            x, y, w, h = cv2.boundingRect(contour)
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w // 2, y + h // 2
            
            # Criar objeto de características
            features = CornGrainFeatures(
                id=i,
                # Geometria
                area=geom_features["area"],
                perimeter=geom_features["perimeter"],
                circularity=geom_features["circularity"],
                aspect_ratio=geom_features["aspect_ratio"],
                solidity=geom_features["solidity"],
                extent=geom_features["extent"],
                equivalent_diameter=geom_features["equivalent_diameter"],
                # Dimensões
                width=w,
                height=h,
                major_axis=geom_features["major_axis"],
                minor_axis=geom_features["minor_axis"],
                # Cor HSV
                mean_hue=color_features["mean_hue"],
                mean_saturation=color_features["mean_saturation"],
                mean_value=color_features["mean_value"],
                std_hue=color_features["std_hue"],
                std_saturation=color_features["std_saturation"],
                std_value=color_features["std_value"],
                # Cor LAB
                mean_l=color_features["mean_l"],
                mean_a=color_features["mean_a"],
                mean_b=color_features["mean_b"],
                # Cor RGB
                mean_r=color_features["mean_r"],
                mean_g=color_features["mean_g"],
                mean_b_rgb=color_features["mean_b_rgb"],
                yellow_index=color_features["yellow_index"],
                # Textura
                texture_contrast=texture_features["contrast"],
                texture_homogeneity=texture_features["homogeneity"],
                texture_energy=texture_features["energy"],
                # Defeitos
                dark_spot_ratio=defect_features["dark_spot_ratio"],
                color_uniformity=defect_features["color_uniformity"],
                # Posição
                centroid=(cx, cy),
                bounding_box=(x, y, w, h),
                defects=[]
            )
            
            self.grains.append(features)
        
        return self.grains
    
    def _extract_geometric_features(self, contour: np.ndarray) -> Dict:
        """Extrai características geométricas"""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (center, axes, angle) = ellipse
            major_axis = max(axes)
            minor_axis = min(axes)
            aspect_ratio = minor_axis / major_axis if major_axis > 0 else 0
        else:
            x, y, w, h = cv2.boundingRect(contour)
            major_axis = max(w, h)
            minor_axis = min(w, h)
            aspect_ratio = minor_axis / major_axis if major_axis > 0 else 0
        
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        x, y, w, h = cv2.boundingRect(contour)
        rect_area = w * h
        extent = area / rect_area if rect_area > 0 else 0
        
        equivalent_diameter = np.sqrt(4 * area / np.pi)
        
        return {
            "area": area,
            "perimeter": perimeter,
            "circularity": circularity,
            "aspect_ratio": aspect_ratio,
            "solidity": solidity,
            "extent": extent,
            "equivalent_diameter": equivalent_diameter,
            "major_axis": major_axis,
            "minor_axis": minor_axis
        }
    
    def _extract_color_features(
        self, image: np.ndarray, hsv: np.ndarray, lab: np.ndarray, mask: np.ndarray
    ) -> Dict:
        """Extrai características de cor incluindo índice de amarelo"""
        hsv_pixels = hsv[mask > 0]
        lab_pixels = lab[mask > 0]
        bgr_pixels = image[mask > 0]
        
        if len(hsv_pixels) == 0:
            return {
                "mean_hue": 0, "mean_saturation": 0, "mean_value": 0,
                "std_hue": 0, "std_saturation": 0, "std_value": 0,
                "mean_l": 0, "mean_a": 0, "mean_b": 0,
                "mean_r": 0, "mean_g": 0, "mean_b_rgb": 0,
                "yellow_index": 0
            }
        
        # HSV
        h, s, v = hsv_pixels[:, 0], hsv_pixels[:, 1], hsv_pixels[:, 2]
        
        # LAB
        l_ch, a_ch, b_ch = lab_pixels[:, 0], lab_pixels[:, 1], lab_pixels[:, 2]
        
        # RGB (BGR -> RGB)
        b_rgb, g_rgb, r_rgb = bgr_pixels[:, 0], bgr_pixels[:, 1], bgr_pixels[:, 2]
        
        # Índice de amarelo: (R + G) / (2 * B)
        # Milho saudável tem alto índice de amarelo
        mean_r = np.mean(r_rgb)
        mean_g = np.mean(g_rgb)
        mean_b_rgb = np.mean(b_rgb)
        yellow_index = (mean_r + mean_g) / (2 * mean_b_rgb + 1)  # +1 para evitar divisão por zero
        
        return {
            "mean_hue": np.mean(h),
            "mean_saturation": np.mean(s),
            "mean_value": np.mean(v),
            "std_hue": np.std(h),
            "std_saturation": np.std(s),
            "std_value": np.std(v),
            "mean_l": np.mean(l_ch),
            "mean_a": np.mean(a_ch),
            "mean_b": np.mean(b_ch),
            "mean_r": mean_r,
            "mean_g": mean_g,
            "mean_b_rgb": mean_b_rgb,
            "yellow_index": yellow_index
        }
    
    def _extract_texture_features(self, gray: np.ndarray, mask: np.ndarray) -> Dict:
        """Extrai características de textura"""
        masked = cv2.bitwise_and(gray, gray, mask=mask)
        
        sobelx = cv2.Sobel(masked, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(masked, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        grain_pixels = gray[mask > 0]
        gradient_pixels = magnitude[mask > 0]
        
        if len(grain_pixels) == 0:
            return {"contrast": 0, "homogeneity": 0, "energy": 0}
        
        contrast = np.var(grain_pixels)
        mean_gradient = np.mean(gradient_pixels)
        homogeneity = 1 / (1 + mean_gradient)
        
        normalized = grain_pixels / 255.0
        energy = np.sum(normalized ** 2) / len(normalized)
        
        return {
            "contrast": contrast,
            "homogeneity": homogeneity,
            "energy": energy
        }
    
    def _extract_defect_features(
        self, image: np.ndarray, hsv: np.ndarray, gray: np.ndarray, mask: np.ndarray
    ) -> Dict:
        """
        Extrai características relacionadas a defeitos em milho.
        """
        grain_pixels = gray[mask > 0]
        hsv_pixels = hsv[mask > 0]
        
        if len(grain_pixels) == 0:
            return {"dark_spot_ratio": 0, "color_uniformity": 1.0}
        
        # Proporção de pixels escuros (potenciais manchas/ardidos)
        dark_threshold = 100  # Pixels mais escuros que isso
        dark_pixels = np.sum(grain_pixels < dark_threshold)
        total_pixels = len(grain_pixels)
        dark_spot_ratio = dark_pixels / total_pixels
        
        # Uniformidade de cor (desvio padrão normalizado)
        color_std = np.std(hsv_pixels[:, 0])  # Variação de matiz
        color_uniformity = 1.0 / (1.0 + color_std / 10.0)
        
        return {
            "dark_spot_ratio": dark_spot_ratio,
            "color_uniformity": color_uniformity
        }
    
    # =========================================================================
    # ETAPA 4: CLASSIFICAÇÃO ESPECÍFICA PARA MILHO
    # =========================================================================
    
    def classify(self, grains: List[CornGrainFeatures]) -> List[CornGrainFeatures]:
        """
        Classifica cada grão de milho baseado nas características.
        
        Defeitos detectados:
        - Ardido: Cor escurecida, baixo valor médio
        - Mofado: Manchas irregulares, baixa uniformidade
        - Fermentado: Alteração de cor (matiz fora do range)
        - Quebrado: Baixa circularidade e solidez
        - Choco: Enrugado, alta variação de textura
        - Descolorido: Baixo índice de amarelo
        """
        for grain in grains:
            defects = self._detect_corn_defects(grain)
            grain.defects = defects
            
            quality, confidence, defect_score = self._calculate_corn_quality(grain, defects)
            grain.quality = quality
            grain.confidence = confidence
            grain.defect_score = defect_score
        
        return grains
    
    def _detect_corn_defects(self, grain: CornGrainFeatures) -> List[CornDefect]:
        """Detecta defeitos específicos do milho"""
        defects = []
        
        # ARDIDO: Grão escurecido por fermentação excessiva
        if grain.mean_value < self.config["ardido_value_threshold"]:
            defects.append(CornDefect.ARDIDO)
        
        # MANCHADO: Presença de manchas escuras
        if grain.dark_spot_ratio > self.config["dark_spot_threshold"]:
            defects.append(CornDefect.MANCHADO)
        
        # FERMENTADO: Cor fora do range normal do milho
        if grain.mean_hue < 10 or grain.mean_hue > 45:
            defects.append(CornDefect.FERMENTADO)
        
        # DESCOLORIDO: Baixo índice de amarelo
        if grain.yellow_index < self.config["yellow_index_min"]:
            defects.append(CornDefect.DESCOLORIDO)
        
        # QUEBRADO: Forma irregular
        if grain.circularity < self.config["circularity_threshold_regular"]:
            if grain.solidity < 0.85:
                defects.append(CornDefect.QUEBRADO)
        
        # CHOCO: Grão imaturo, enrugado
        if grain.texture_contrast > 800 and grain.mean_saturation < 100:
            defects.append(CornDefect.CHOCO)
        
        # MOFADO: Baixa uniformidade de cor com manchas
        if grain.color_uniformity < 0.7 and grain.std_value > 30:
            defects.append(CornDefect.MOFADO)
        
        # CARUNCHADO: Baixa solidez (buracos)
        if grain.solidity < 0.80 and grain.extent < 0.6:
            defects.append(CornDefect.CARUNCHADO)
        
        if not defects:
            defects.append(CornDefect.NENHUM)
        
        return defects
    
    def _calculate_corn_quality(
        self, grain: CornGrainFeatures, defects: List[CornDefect]
    ) -> Tuple[CornQuality, float, float]:
        """
        Calcula a qualidade do grão de milho.
        
        Returns:
            Tupla (qualidade, confiança, score_defeito)
        """
        score = 100.0
        
        # Penalidades por defeitos (pesos baseados em gravidade)
        defect_penalties = {
            CornDefect.NENHUM: 0,
            CornDefect.ARDIDO: 35,          # Grave
            CornDefect.MOFADO: 40,           # Muito grave
            CornDefect.FERMENTADO: 25,       # Moderado
            CornDefect.GERMINADO: 20,        # Moderado
            CornDefect.CARUNCHADO: 35,       # Grave
            CornDefect.CHOCO: 30,            # Grave
            CornDefect.QUEBRADO: 15,         # Leve
            CornDefect.DESCOLORIDO: 15,      # Leve
            CornDefect.MANCHADO: 20,         # Moderado
        }
        
        for defect in defects:
            score -= defect_penalties.get(defect, 10)
        
        # Penalidade por forma (milho deve ser oval)
        optimal_ar = 0.75  # Aspect ratio ideal do milho
        ar_deviation = abs(grain.aspect_ratio - optimal_ar)
        score -= ar_deviation * 15
        
        # Penalidade por cor fora do ideal
        optimal_hue = 25  # Amarelo ideal
        hue_deviation = abs(grain.mean_hue - optimal_hue)
        if hue_deviation > 10:
            score -= (hue_deviation - 10) * 0.5
        
        # Normalizar score
        score = max(0, min(100, score))
        confidence = score / 100.0
        
        # Calcular score de defeito (0-1)
        num_defects = len([d for d in defects if d != CornDefect.NENHUM])
        defect_score = num_defects / 5.0  # Normalizado para max 5 defeitos
        defect_score = min(1.0, defect_score)
        
        # Determinar tipo/categoria
        if score >= 90 and defect_score <= self.config["tipo1_max_defects"]:
            quality = CornQuality.TIPO_1
        elif score >= 75 and defect_score <= self.config["tipo2_max_defects"]:
            quality = CornQuality.TIPO_2
        elif score >= 55 and defect_score <= self.config["tipo3_max_defects"]:
            quality = CornQuality.TIPO_3
        elif score >= 35:
            quality = CornQuality.FORA_TIPO
        else:
            quality = CornQuality.DESCARTE
        
        return quality, confidence, defect_score
    
    # =========================================================================
    # VISUALIZAÇÃO E RELATÓRIOS
    # =========================================================================
    
    def visualize(
        self, image: np.ndarray, contours: List[np.ndarray], grains: List[CornGrainFeatures]
    ) -> np.ndarray:
        """
        Cria visualização dos resultados da classificação.
        """
        result = image.copy()
        
        # Cores por qualidade
        colors = {
            CornQuality.TIPO_1: (0, 255, 0),      # Verde
            CornQuality.TIPO_2: (0, 200, 100),    # Verde claro
            CornQuality.TIPO_3: (0, 255, 255),    # Amarelo
            CornQuality.FORA_TIPO: (0, 165, 255), # Laranja
            CornQuality.DESCARTE: (0, 0, 255)     # Vermelho
        }
        
        for contour, grain in zip(contours, grains):
            color = colors.get(grain.quality, (255, 255, 255))
            
            # Desenhar contorno
            cv2.drawContours(result, [contour], -1, color, 2)
            
            # Adicionar label
            cx, cy = grain.centroid
            # Abreviação da qualidade
            qual_abbrev = {
                CornQuality.TIPO_1: "T1",
                CornQuality.TIPO_2: "T2",
                CornQuality.TIPO_3: "T3",
                CornQuality.FORA_TIPO: "FT",
                CornQuality.DESCARTE: "X"
            }
            label = f"{grain.id}:{qual_abbrev.get(grain.quality, '?')}"
            cv2.putText(
                result, label, (cx - 15, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
            )
        
        # Adicionar legenda
        legend_y = 20
        for quality, color in colors.items():
            cv2.rectangle(result, (10, legend_y - 12), (25, legend_y + 2), color, -1)
            cv2.putText(result, quality.value, (30, legend_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            legend_y += 20
        
        return result
    
    def generate_report(self, grains: List[CornGrainFeatures]) -> Dict:
        """Gera relatório estatístico da classificação de milho"""
        if not grains:
            return {"error": "Nenhum grão de milho detectado"}
        
        # Contagem por qualidade
        quality_counts = {}
        for quality in CornQuality:
            quality_counts[quality.value] = sum(
                1 for g in grains if g.quality == quality
            )
        
        # Estatísticas
        areas = [g.area for g in grains]
        circularities = [g.circularity for g in grains]
        yellow_indices = [g.yellow_index for g in grains]
        
        report = {
            "tipo_grao": "MILHO",
            "total_graos": len(grains),
            "classificacao": quality_counts,
            "percentuais": {
                k: round(v / len(grains) * 100, 1)
                for k, v in quality_counts.items()
            },
            "estatisticas": {
                "area_media": round(np.mean(areas), 2),
                "area_std": round(np.std(areas), 2),
                "circularidade_media": round(np.mean(circularities), 3),
                "indice_amarelo_medio": round(np.mean(yellow_indices), 2),
            },
            "defeitos_encontrados": {},
            "graos": []
        }
        
        # Contagem de defeitos
        for grain in grains:
            for defect in grain.defects:
                if defect != CornDefect.NENHUM:
                    name = defect.value
                    report["defeitos_encontrados"][name] = \
                        report["defeitos_encontrados"].get(name, 0) + 1
        
        # Detalhes por grão
        for grain in grains:
            report["graos"].append({
                "id": grain.id,
                "qualidade": grain.quality.value,
                "confianca": round(grain.confidence, 2),
                "score_defeito": round(grain.defect_score, 3),
                "circularidade": round(grain.circularity, 3),
                "aspect_ratio": round(grain.aspect_ratio, 3),
                "indice_amarelo": round(grain.yellow_index, 2),
                "area": round(grain.area, 1),
                "defeitos": [d.value for d in grain.defects]
            })
        
        # Resumo para classificação do lote
        total = len(grains)
        tipo1_pct = quality_counts.get(CornQuality.TIPO_1.value, 0) / total * 100
        tipo2_pct = quality_counts.get(CornQuality.TIPO_2.value, 0) / total * 100
        defeituosos = quality_counts.get(CornQuality.FORA_TIPO.value, 0) + \
                      quality_counts.get(CornQuality.DESCARTE.value, 0)
        defeituosos_pct = defeituosos / total * 100
        
        if defeituosos_pct <= 1:
            lote_class = "TIPO 1"
        elif defeituosos_pct <= 2:
            lote_class = "TIPO 2"
        elif defeituosos_pct <= 3:
            lote_class = "TIPO 3"
        else:
            lote_class = "FORA DE TIPO"
        
        report["classificacao_lote"] = {
            "tipo": lote_class,
            "percentual_defeituosos": round(defeituosos_pct, 2)
        }
        
        return report
    
    # =========================================================================
    # PIPELINE COMPLETA
    # =========================================================================
    
    def process(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Executa a pipeline completa de classificação de milho.
        
        Args:
            image: Imagem BGR de entrada
            
        Returns:
            Tupla com (imagem anotada, relatório)
        """
        # Etapa 1: Pré-processamento
        preprocessed = self.preprocess(image)
        
        # Etapa 2: Segmentação
        mask, contours = self.segment(preprocessed)
        
        # Etapa 3: Extração de características
        grains = self.extract_features(self.original_image, contours)
        
        # Etapa 4: Classificação
        classified = self.classify(grains)
        
        # Gerar visualização e relatório
        annotated = self.visualize(self.original_image, contours, classified)
        report = self.generate_report(classified)
        
        return annotated, report


def main():
    """Função principal para demonstração"""
    import sys
    
    if len(sys.argv) < 2:
        print("=" * 60)
        print("Pipeline de Classificação de Grãos de MILHO")
        print("=" * 60)
        print("\nUso: python corn_classifier.py <caminho_imagem>")
        print("\nExemplo:")
        print("  python corn_classifier.py milho.jpg")
        print("\nDefeitos detectados:")
        print("  - Ardido (fermentação)")
        print("  - Mofado")
        print("  - Fermentado")
        print("  - Quebrado")
        print("  - Choco (imaturo)")
        print("  - Descolorido")
        print("  - Manchado")
        print("  - Carunchado")
        return
    
    image_path = sys.argv[1]
    
    # Carregar imagem
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erro: Não foi possível carregar a imagem '{image_path}'")
        return
    
    print(f"Imagem carregada: {image.shape}")
    
    # Criar pipeline
    pipeline = CornClassifierPipeline()
    
    # Processar
    print("\nProcessando grãos de milho...")
    annotated, report = pipeline.process(image)
    
    # Exibir resultados
    print("\n" + "=" * 60)
    print("RELATÓRIO DE CLASSIFICAÇÃO - MILHO")
    print("=" * 60)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    
    # Salvar imagem
    output_path = "resultado_milho.png"
    cv2.imwrite(output_path, annotated)
    print(f"\nImagem salva em: {output_path}")


if __name__ == "__main__":
    main()
