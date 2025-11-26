"""
Pipeline de Classificação de Grãos usando OpenCV
=================================================

Este script implementa uma pipeline completa de processamento de imagem
para segmentar, extrair características e classificar grãos de soja.

Etapas da Pipeline:
1. Pré-processamento (redução de ruído, normalização)
2. Segmentação (detecção e separação de grãos individuais)
3. Extração de características (cor, forma, textura)
4. Classificação baseada em regras

Autor: Pipeline gerada com Claude
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from enum import Enum
import json
import os


class GrainQuality(Enum):
    """Classificação de qualidade dos grãos"""
    EXCELENTE = "Excelente"
    BOM = "Bom"
    REGULAR = "Regular"
    DEFEITUOSO = "Defeituoso"
    DANIFICADO = "Danificado"


class GrainDefect(Enum):
    """Tipos de defeitos detectáveis"""
    NENHUM = "Nenhum"
    MANCHA_ESCURA = "Mancha Escura"
    DEFORMACAO = "Deformação"
    RACHADURA = "Rachadura"
    DESCOLORACAO = "Descoloração"
    ENRUGAMENTO = "Enrugamento"


@dataclass
class GrainFeatures:
    """Características extraídas de cada grão"""
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
    
    # Cor (espaço HSV)
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
    
    # Textura
    texture_contrast: float
    texture_homogeneity: float
    texture_energy: float
    
    # Posição
    centroid: Tuple[int, int]
    bounding_box: Tuple[int, int, int, int]
    
    # Classificação
    quality: GrainQuality = GrainQuality.BOM
    defects: List[GrainDefect] = None
    confidence: float = 0.0


class GrainClassifierPipeline:
    """
    Pipeline completa para classificação de grãos usando processamento de imagem.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Inicializa a pipeline com configurações personalizáveis.
        
        Args:
            config: Dicionário com parâmetros de configuração
        """
        self.config = config or self._default_config()
        self.grains: List[GrainFeatures] = []
        self.original_image = None
        self.processed_image = None
        self.mask = None
        
    def _default_config(self) -> Dict:
        """Retorna configurações padrão da pipeline"""
        return {
            # Pré-processamento
            "blur_kernel_size": 5,
            "bilateral_d": 9,
            "bilateral_sigma_color": 75,
            "bilateral_sigma_space": 75,
            
            # Segmentação
            "threshold_method": "combined",  # "otsu", "adaptive", "color", "combined"
            "morph_kernel_size": 3,
            "min_grain_area": 100,  # Reduzido para grãos menores
            "max_grain_area": 100000,
            
            # Watershed
            "use_watershed": True,
            "dist_transform_mask": 5,
            "watershed_threshold": 0.4,
            
            # Classificação
            "circularity_threshold_excellent": 0.85,
            "circularity_threshold_good": 0.75,
            "circularity_threshold_regular": 0.60,
            
            "aspect_ratio_min": 0.7,
            "aspect_ratio_max": 1.4,
            
            "dark_spot_threshold": 0.3,
            "color_variation_threshold": 30,
        }


    
    # =========================================================================
    # ETAPA 1: PRÉ-PROCESSAMENTO
    # =========================================================================
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Aplica pré-processamento à imagem para melhorar a segmentação.
        
        Técnicas aplicadas:
        - Redução de ruído com filtro bilateral (preserva bordas)
        - Ajuste de contraste com CLAHE
        - Suavização gaussiana
        
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
        
        # 3. Suavização leve
        kernel_size = self.config["blur_kernel_size"]
        smoothed = cv2.GaussianBlur(enhanced, (kernel_size, kernel_size), 0)
        
        self.processed_image = smoothed
        return smoothed
    
    # =========================================================================
    # ETAPA 2: SEGMENTAÇÃO
    # =========================================================================
    
    def segment(self, image: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Segmenta a imagem para identificar grãos individuais.
        
        Métodos disponíveis:
        - Otsu: Binarização automática
        - Adaptive: Threshold adaptativo
        - Color: Segmentação por cor no espaço HSV
        
        Args:
            image: Imagem pré-processada
            
        Returns:
            Máscara binária e lista de contornos
        """
        method = self.config["threshold_method"]
        
        if method == "otsu":
            mask = self._segment_otsu(image)
        elif method == "adaptive":
            mask = self._segment_adaptive(image)
        elif method == "color":
            mask = self._segment_color(image)
        elif method == "combined":
            mask = self._segment_combined(image)
        else:
            mask = self._segment_combined(image)
        
        # Aplicar operações morfológicas para limpar a máscara
        mask = self._morphological_cleanup(mask)
        
        # Usar watershed se configurado
        if self.config["use_watershed"]:
            mask = self._apply_watershed(image, mask)
        
        self.mask = mask
        
        # Encontrar contornos
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filtrar contornos por área
        min_area = self.config["min_grain_area"]
        max_area = self.config["max_grain_area"]
        
        valid_contours = [
            cnt for cnt in contours
            if min_area < cv2.contourArea(cnt) < max_area
        ]
        
        # Se poucos contornos foram encontrados mas a máscara tem área significativa,
        # tentar detecção por Hough Circles (grãos muito próximos)
        total_mask_area = np.sum(mask > 0)
        if len(valid_contours) <= 2 and total_mask_area > 5000:
            hough_contours = self._detect_grains_hough(self.processed_image, mask)
            if len(hough_contours) > len(valid_contours):
                valid_contours = hough_contours
        
        return mask, valid_contours
    
    def _segment_otsu(self, image: np.ndarray) -> np.ndarray:
        """Segmentação usando threshold de Otsu"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return mask
    
    def _segment_adaptive(self, image: np.ndarray) -> np.ndarray:
        """Segmentação usando threshold adaptativo"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        return mask
    
    def _segment_color(self, image: np.ndarray) -> np.ndarray:
        """Segmentação baseada em cor (específica para soja)"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Range de cor para grãos de soja (amarelo/bege)
        lower = np.array([10, 20, 100])
        upper = np.array([40, 255, 255])
        
        mask = cv2.inRange(hsv, lower, upper)
        return mask
    
    def _segment_combined(self, image: np.ndarray) -> np.ndarray:
        """
        Segmentação combinada: usa múltiplas técnicas e combina resultados.
        Ideal para fundo branco com grãos coloridos.
        """
        # Converter para diferentes espaços
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Método 1: Otsu invertido (para fundo claro)
        _, mask_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Método 2: Segmentação por saturação (grãos têm mais saturação que fundo branco)
        saturation = hsv[:, :, 1]
        _, mask_sat = cv2.threshold(saturation, 30, 255, cv2.THRESH_BINARY)
        
        # Método 3: Segmentação por cor no espaço LAB
        # Canal 'b' diferencia amarelo/bege de branco
        b_channel = lab[:, :, 2]
        _, mask_lab = cv2.threshold(b_channel, 135, 255, cv2.THRESH_BINARY)
        
        # Método 4: Range HSV específico para soja
        lower_hsv = np.array([10, 15, 80])
        upper_hsv = np.array([45, 255, 255])
        mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
        
        # Combinar máscaras (interseção das técnicas mais confiáveis)
        # Usa AND entre métodos para reduzir falsos positivos
        mask_combined = cv2.bitwise_and(mask_otsu, mask_sat)
        mask_combined = cv2.bitwise_or(mask_combined, mask_hsv)
        
        return mask_combined
    
    def _detect_grains_hough(self, image: np.ndarray, mask: np.ndarray) -> List[np.ndarray]:
        """
        Detecta grãos usando Hough Circle Transform.
        Útil quando grãos estão muito próximos.
        
        Args:
            image: Imagem original
            mask: Máscara de segmentação
            
        Returns:
            Lista de contornos aproximados como círculos
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Detectar círculos
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=12,
            param1=40,
            param2=20,
            minRadius=6,
            maxRadius=35
        )
        
        contours = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                x, y, r = circle
                # Verificar se o centro está dentro da máscara
                if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                    if mask[y, x] > 0:
                        # Criar contorno circular
                        angles = np.linspace(0, 2 * np.pi, 36)
                        pts = np.array([
                            [int(x + r * np.cos(a)), int(y + r * np.sin(a))]
                            for a in angles
                        ], dtype=np.int32)
                        contours.append(pts.reshape(-1, 1, 2))
        
        return contours
    
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
        Aplica algoritmo watershed para separar grãos conectados.
        
        O watershed é especialmente útil quando grãos estão se tocando.
        """
        # Distance transform
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, self.config["dist_transform_mask"])
        
        # Normalizar
        dist_normalized = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
        
        # Aplicar threshold dinâmico baseado no tamanho dos blobs
        # Threshold mais baixo para imagens com muitos grãos juntos
        threshold = self.config["watershed_threshold"]
        
        _, sure_fg = cv2.threshold(
            dist_normalized, threshold, 1.0, cv2.THRESH_BINARY
        )
        sure_fg = np.uint8(sure_fg * 255)
        
        # Se muito poucos foreground pixels, tentar threshold menor
        if np.sum(sure_fg > 0) < 100:
            _, sure_fg = cv2.threshold(
                dist_normalized, threshold * 0.5, 1.0, cv2.THRESH_BINARY
            )
            sure_fg = np.uint8(sure_fg * 255)
        
        # Encontrar área de fundo certeza
        kernel = np.ones((3, 3), np.uint8)
        sure_bg = cv2.dilate(mask, kernel, iterations=3)
        
        # Região desconhecida
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Rotular marcadores
        num_labels, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # Aplicar watershed
        image_bgr = image if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(image_bgr, markers)
        
        # Criar máscara final
        result_mask = np.zeros_like(mask)
        result_mask[markers > 1] = 255
        
        return result_mask
    
    # =========================================================================
    # ETAPA 3: EXTRAÇÃO DE CARACTERÍSTICAS
    # =========================================================================
    
    def extract_features(
        self, image: np.ndarray, contours: List[np.ndarray]
    ) -> List[GrainFeatures]:
        """
        Extrai características de cada grão segmentado.
        
        Características extraídas:
        - Geométricas: área, perímetro, circularidade, etc.
        - Cor: médias e desvios em HSV e LAB
        - Textura: baseadas em GLCM simplificado
        
        Args:
            image: Imagem original
            contours: Lista de contornos dos grãos
            
        Returns:
            Lista de objetos GrainFeatures
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
            
            # Extrair características geométricas
            geom_features = self._extract_geometric_features(contour)
            
            # Extrair características de cor
            color_features = self._extract_color_features(hsv, lab, grain_mask)
            
            # Extrair características de textura
            texture_features = self._extract_texture_features(gray, grain_mask)
            
            # Obter bounding box e centróide
            x, y, w, h = cv2.boundingRect(contour)
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w // 2, y + h // 2
            
            # Criar objeto de características
            features = GrainFeatures(
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
                # Textura
                texture_contrast=texture_features["contrast"],
                texture_homogeneity=texture_features["homogeneity"],
                texture_energy=texture_features["energy"],
                # Posição
                centroid=(cx, cy),
                bounding_box=(x, y, w, h),
                defects=[]
            )
            
            self.grains.append(features)
        
        return self.grains
    
    def _extract_geometric_features(self, contour: np.ndarray) -> Dict:
        """Extrai características geométricas do contorno"""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Circularidade: 4π * área / perímetro²
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        # Ajustar elipse se tiver pontos suficientes
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
        
        # Convex hull para solidez
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Extent: área do contorno / área do bounding rect
        x, y, w, h = cv2.boundingRect(contour)
        rect_area = w * h
        extent = area / rect_area if rect_area > 0 else 0
        
        # Diâmetro equivalente
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
        self, hsv: np.ndarray, lab: np.ndarray, mask: np.ndarray
    ) -> Dict:
        """Extrai características de cor nos espaços HSV e LAB"""
        # Extrair pixels do grão
        hsv_pixels = hsv[mask > 0]
        lab_pixels = lab[mask > 0]
        
        if len(hsv_pixels) == 0:
            return {
                "mean_hue": 0, "mean_saturation": 0, "mean_value": 0,
                "std_hue": 0, "std_saturation": 0, "std_value": 0,
                "mean_l": 0, "mean_a": 0, "mean_b": 0
            }
        
        # Estatísticas HSV
        h, s, v = hsv_pixels[:, 0], hsv_pixels[:, 1], hsv_pixels[:, 2]
        
        # Estatísticas LAB
        l_ch, a_ch, b_ch = lab_pixels[:, 0], lab_pixels[:, 1], lab_pixels[:, 2]
        
        return {
            "mean_hue": np.mean(h),
            "mean_saturation": np.mean(s),
            "mean_value": np.mean(v),
            "std_hue": np.std(h),
            "std_saturation": np.std(s),
            "std_value": np.std(v),
            "mean_l": np.mean(l_ch),
            "mean_a": np.mean(a_ch),
            "mean_b": np.mean(b_ch)
        }
    
    def _extract_texture_features(
        self, gray: np.ndarray, mask: np.ndarray
    ) -> Dict:
        """
        Extrai características de textura simplificadas.
        
        Usa gradientes locais como aproximação de GLCM.
        """
        # Aplicar máscara
        masked = cv2.bitwise_and(gray, gray, mask=mask)
        
        # Calcular gradientes
        sobelx = cv2.Sobel(masked, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(masked, cv2.CV_64F, 0, 1, ksize=3)
        
        # Magnitude do gradiente
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Extrair apenas pixels do grão
        grain_pixels = gray[mask > 0]
        gradient_pixels = magnitude[mask > 0]
        
        if len(grain_pixels) == 0:
            return {"contrast": 0, "homogeneity": 0, "energy": 0}
        
        # Contraste: variância da intensidade
        contrast = np.var(grain_pixels)
        
        # Homogeneidade: inverso da média do gradiente
        mean_gradient = np.mean(gradient_pixels)
        homogeneity = 1 / (1 + mean_gradient)
        
        # Energia: soma dos quadrados normalizados
        normalized = grain_pixels / 255.0
        energy = np.sum(normalized ** 2) / len(normalized)
        
        return {
            "contrast": contrast,
            "homogeneity": homogeneity,
            "energy": energy
        }
    
    # =========================================================================
    # ETAPA 4: CLASSIFICAÇÃO
    # =========================================================================
    
    def classify(self, grains: List[GrainFeatures]) -> List[GrainFeatures]:
        """
        Classifica cada grão baseado nas características extraídas.
        
        Sistema de classificação baseado em regras:
        - Excelente: Formato ideal, cor uniforme, sem defeitos
        - Bom: Pequenos desvios, ainda aceitável
        - Regular: Desvios notáveis, uso limitado
        - Defeituoso: Problemas significativos
        - Danificado: Grão comprometido
        
        Args:
            grains: Lista de características dos grãos
            
        Returns:
            Lista de grãos com classificação atualizada
        """
        for grain in grains:
            # Detectar defeitos
            defects = self._detect_defects(grain)
            grain.defects = defects
            
            # Calcular score de qualidade
            quality, confidence = self._calculate_quality(grain, defects)
            grain.quality = quality
            grain.confidence = confidence
        
        return grains
    
    def _detect_defects(self, grain: GrainFeatures) -> List[GrainDefect]:
        """Detecta defeitos no grão baseado nas características"""
        defects = []
        
        # Verificar manchas escuras (baixo valor médio)
        if grain.mean_value < 100:
            defects.append(GrainDefect.MANCHA_ESCURA)
        
        # Verificar deformação (baixa circularidade)
        if grain.circularity < self.config["circularity_threshold_regular"]:
            defects.append(GrainDefect.DEFORMACAO)
        
        # Verificar aspect ratio anormal
        if (grain.aspect_ratio < self.config["aspect_ratio_min"] or 
            grain.aspect_ratio > self.config["aspect_ratio_max"]):
            defects.append(GrainDefect.DEFORMACAO)
        
        # Verificar descoloração (alta variação de cor)
        if grain.std_value > self.config["color_variation_threshold"]:
            defects.append(GrainDefect.DESCOLORACAO)
        
        # Verificar enrugamento (alta variação de textura)
        if grain.texture_contrast > 1000:
            defects.append(GrainDefect.ENRUGAMENTO)
        
        if not defects:
            defects.append(GrainDefect.NENHUM)
        
        return defects
    
    def _calculate_quality(
        self, grain: GrainFeatures, defects: List[GrainDefect]
    ) -> Tuple[GrainQuality, float]:
        """Calcula a qualidade final do grão"""
        score = 100.0
        
        # Penalizar por circularidade
        if grain.circularity >= self.config["circularity_threshold_excellent"]:
            score -= 0
        elif grain.circularity >= self.config["circularity_threshold_good"]:
            score -= 10
        elif grain.circularity >= self.config["circularity_threshold_regular"]:
            score -= 25
        else:
            score -= 40
        
        # Penalizar por aspect ratio
        optimal_ar = 1.0
        ar_deviation = abs(grain.aspect_ratio - optimal_ar)
        score -= ar_deviation * 20
        
        # Penalizar por defeitos
        defect_penalties = {
            GrainDefect.NENHUM: 0,
            GrainDefect.MANCHA_ESCURA: 20,
            GrainDefect.DEFORMACAO: 15,
            GrainDefect.RACHADURA: 25,
            GrainDefect.DESCOLORACAO: 10,
            GrainDefect.ENRUGAMENTO: 15
        }
        
        for defect in defects:
            score -= defect_penalties.get(defect, 10)
        
        # Normalizar score
        score = max(0, min(100, score))
        confidence = score / 100.0
        
        # Determinar categoria
        if score >= 85:
            quality = GrainQuality.EXCELENTE
        elif score >= 70:
            quality = GrainQuality.BOM
        elif score >= 50:
            quality = GrainQuality.REGULAR
        elif score >= 30:
            quality = GrainQuality.DEFEITUOSO
        else:
            quality = GrainQuality.DANIFICADO
        
        return quality, confidence
    
    # =========================================================================
    # VISUALIZAÇÃO E EXPORTAÇÃO
    # =========================================================================
    
    def visualize(
        self, image: np.ndarray, contours: List[np.ndarray], grains: List[GrainFeatures]
    ) -> np.ndarray:
        """
        Cria visualização dos resultados da classificação.
        
        Args:
            image: Imagem original
            contours: Lista de contornos
            grains: Lista de características classificadas
            
        Returns:
            Imagem com anotações
        """
        result = image.copy()
        
        # Cores por qualidade
        colors = {
            GrainQuality.EXCELENTE: (0, 255, 0),    # Verde
            GrainQuality.BOM: (0, 200, 100),         # Verde claro
            GrainQuality.REGULAR: (0, 255, 255),     # Amarelo
            GrainQuality.DEFEITUOSO: (0, 165, 255), # Laranja
            GrainQuality.DANIFICADO: (0, 0, 255)    # Vermelho
        }
        
        for contour, grain in zip(contours, grains):
            color = colors.get(grain.quality, (255, 255, 255))
            
            # Desenhar contorno
            cv2.drawContours(result, [contour], -1, color, 2)
            
            # Adicionar texto
            cx, cy = grain.centroid
            label = f"{grain.id}: {grain.quality.value[:3]}"
            cv2.putText(
                result, label, (cx - 20, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
            )
        
        return result
    
    def generate_report(self, grains: List[GrainFeatures]) -> Dict:
        """Gera relatório estatístico da classificação"""
        if not grains:
            return {"error": "Nenhum grão detectado"}
        
        # Contagem por qualidade
        quality_counts = {}
        for quality in GrainQuality:
            quality_counts[quality.value] = sum(
                1 for g in grains if g.quality == quality
            )
        
        # Estatísticas gerais
        areas = [g.area for g in grains]
        circularities = [g.circularity for g in grains]
        
        report = {
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
                "circularidade_std": round(np.std(circularities), 3)
            },
            "defeitos_encontrados": {},
            "graos": []
        }
        
        # Contagem de defeitos
        for grain in grains:
            for defect in grain.defects:
                if defect != GrainDefect.NENHUM:
                    name = defect.value
                    report["defeitos_encontrados"][name] = \
                        report["defeitos_encontrados"].get(name, 0) + 1
        
        # Detalhes de cada grão
        for grain in grains:
            report["graos"].append({
                "id": grain.id,
                "qualidade": grain.quality.value,
                "confianca": round(grain.confidence, 2),
                "circularidade": round(grain.circularity, 3),
                "aspect_ratio": round(grain.aspect_ratio, 3),
                "area": round(grain.area, 1),
                "defeitos": [d.value for d in grain.defects]
            })
        
        return report
    
    # =========================================================================
    # PIPELINE COMPLETA
    # =========================================================================
    
    def process(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Executa a pipeline completa de classificação.
        
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
    
    # Verificar argumentos
    if len(sys.argv) < 2:
        print("Uso: python grain_classifier.py <caminho_imagem>")
        print("\nExemplo:")
        print("  python grain_classifier.py soja.jpg")
        return
    
    image_path = sys.argv[1]
    
    # Carregar imagem
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erro: Não foi possível carregar a imagem '{image_path}'")
        return
    
    print(f"Imagem carregada: {image.shape}")
    
    # Criar pipeline
    pipeline = GrainClassifierPipeline()
    
    # Processar
    print("\nProcessando...")
    annotated, report = pipeline.process(image)
    
    # Exibir resultados
    print("\n" + "=" * 50)
    print("RELATÓRIO DE CLASSIFICAÇÃO")
    print("=" * 50)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    
    # Salvar imagem anotada
    output_path = "resultado_classificacao.png"
    cv2.imwrite(output_path, annotated)
    print(f"\nImagem salva em: {output_path}")


if __name__ == "__main__":
    main()
