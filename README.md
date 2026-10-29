# Kinetix — 3D Computer Vision Biomechanics & Kinematics Engine

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0+-5C3EE8.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose%203D-0097A7.svg?logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![Tests Passing](https://img.shields.io/badge/Pytest-25%20passed%20(100%25)-16A34A.svg?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![Docker Ready](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-gray.svg)](https://opensource.org/licenses/MIT)

> **Motor clínico de visión artificial y cinemática articular 3D para evaluación técnica en sentadillas (squats) y flexiones de pecho (push-ups) desde video monocular estándar.**  
> Desarrollado por **Nicolás Cardozo** (*Estudiante de Ciencia de Datos, Pontificia Universidad Javeriana, Bogotá*).

<div align="center">
  <br>
  <img src="outputs/annotated_squat_analysis.gif" alt="Análisis Biomecánico con Visión Artificial" width="380px" style="border-radius: 8px; border: 1px solid #E4E4E7;" />
  <p><em>Trazado articular en tiempo real: flexión de rodilla (ROM), inclinación espinal, colinealidad de plancha y cadencia.</em></p>
  <br>
</div>

---

## Resumen Ejecutivo y Formulación del Problema

En el entrenamiento de fuerza, la preparación atlética y la rehabilitación musculoesquelética, las desviaciones cinemáticas determinan tanto el rendimiento como el riesgo de lesión:

* **Colapso en Valgo Dinámico de Rodilla (FPPA):** Principal causa biomecánica de roturas del ligamento cruzado anterior (LCA) y síndrome patelofemoral durante la fase concéntrica de la sentadilla.
* **Inclinación Excesiva de Tronco ($> 45^\circ$):** Incrementa exponencialmente las fuerzas de cizallamiento sobre las vértebras lumbares L4-L5 y L5-S1.
* **Profundidad Articular Insuficiente:** Limita el reclutamiento funcional de glúteos y cuádriceps mientras concentra el estrés en el tendón rotuliano.
* **Rango de Movimiento en Flexiones (Codo $> 90^\circ$):** Repeticiones parciales que incumplen estándares militares y clínicos, reduciendo la activación del pectoral mayor.
* **Inestabilidad de Plancha / Hip Sag:** Hiperextensión lumbar bajo carga gravitacional debida a fatiga o falta de activación del core.

Kinetix automatiza la auditoría de estos movimientos combinando **MediaPipe Tasks Vision**, **cálculo vectorial 3D**, **filtrado polinomial de Savitzky-Golay**, una **API REST en FastAPI** y un **dashboard interactivo** de alta legibilidad.

---

## Fundamentos Matemáticos y Cinemática

### 1. Cálculo Vectorial de Ángulos Articulares 3D
Dados tres puntos anatómicos $\mathbf{a}$ (proximal), $\mathbf{b}$ (vértice articular) y $\mathbf{c}$ (distal) en $\mathbb{R}^3$:

$$\mathbf{u} = \mathbf{a} - \mathbf{b}, \quad \mathbf{v} = \mathbf{c} - \mathbf{b}$$

$$\theta_{\text{articular}} = \arccos\left(\text{clamp}\left(\frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}, -1.0, 1.0\right)\right) \times \frac{180}{\pi}$$

### 2. Corrección Isotrópica para Video Vertical (9:16)
Los videos grabados en celulares con orientación vertical deforman los ángulos si se usan coordenadas normalizadas $[0, 1]$. El pipeline proyecta las coordenadas al espacio euclidiano métrico en píxeles:

$$p_x = x \cdot W, \quad p_y = y \cdot H$$

### 3. Cinemática de Flexiones de Pecho (Push-Ups)
* **Flexión de Codo ($\theta_{\text{codo}}$):** Formada por Hombro $\to$ Codo $\to$ Muñeca. Bloqueo articular superior: $\ge 160^\circ$; Profundidad estándar militar / clínica: $\le 90^\circ$.
* **Alineación de Plancha ($\theta_{\text{plancha}}$):** Colinealidad Hombro $\to$ Cadera $\to$ Tobillo. Rango neutro ideal: $170^\circ - 180^\circ$. Ángulos $< 160^\circ$ activan alerta de sobrecarga lumbar (*Hip Sag*).

### 4. Valgo Dinámico de Rodilla (FPPA)
Cuantifica la desviación medial de la rótula respecto al eje mecánico que conecta la cadera con el tobillo en el plano frontal:

$$\hat{x}_{\text{rodilla}} = x_{\text{cadera}} + \frac{y_{\text{rodilla}} - y_{\text{cadera}}}{y_{\text{tobillo}} - y_{\text{cadera}} + 10^{-7}} (x_{\text{tobillo}} - x_{\text{cadera}})$$

$$\delta_{\text{valgo}} = (x_{\text{rodilla}} - \hat{x}_{\text{rodilla}}) \times \text{signo}_{\text{miembro}}$$

### 5. Filtrado de Señal Savitzky-Golay
Elimina el temblor de alta frecuencia de grabaciones manuales sin retrasar la fase ni aplanar los puntos críticos de inflexión:

$$\hat{y}_k = \sum_{i=-m}^m c_i y_{k+i}$$

---

## Demostración y Prueba Rápida

### 1. Demos de 1 Clic Integradas
El dashboard web incluye pruebas precargadas con telemetría completa:
* **Demo Sentadilla:** Evalúa un ciclo completo con análisis de profundidad ROM, inclinación espinal y estabilidad de rodillas.
* **Demo Flexión (Push-Up):** Evalúa flexiones con detección de codo a 90° y colinealidad de la plancha dorsal.
* **Subir Video:** Permite cargar cualquier video grabado con celular; el sistema detecta automáticamente si el ejercicio es sentadilla o flexión.

### 2. Despliegue en la Nube
El proyecto incluye un `Dockerfile` optimizado y una guía en [**DEPLOYMENT.md**](DEPLOYMENT.md) para despliegue permanente en servicios como Hugging Face Spaces o Railway.

---

## Instalación y Uso Local

```bash
# 1. Clonar repositorio
git clone https://github.com/nicolascardozo0709/biomechanics-computer-vision.git
cd biomechanics-computer-vision

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar análisis por consola
python main.py --benchmark --exercise pushup
python main.py --benchmark --exercise squat
python main.py --source data/tu_video.mp4

# 4. Iniciar la aplicación web
python app.py
```
Abre en tu navegador: **`http://127.0.0.1:8000`**

### Pruebas Automatizadas
```bash
pytest tests/ -v
```
*(25 de 25 pruebas unitarias y de integración pasando).*

---

## Endpoints de la API RESTful (FastAPI)

| Método | Endpoint | Parámetros | Descripción |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | — | Dashboard web interactivo para análisis biomecánico |
| `GET` | `/api/health` | — | Estado del servicio y disponibilidad del modelo MediaPipe |
| `POST` | `/api/load_benchmark` | `exercise=squat\|pushup` | Ejecuta benchmark cinemático y actualiza el reporte activo |
| `POST` | `/api/analyze` | `file` (video), `exercise=auto\|squat\|pushup` | Ingesta de video y auditoría cinemática 3D completa |
| `GET` | `/api/artifacts/{filename}` | — | Sirve videos anotados MP4/GIF, gráficos de telemetría y JSON |
| `GET` | `/docs` | — | Documentación interactiva Swagger / OpenAPI |

---

## Autor & Contacto

* **Nicolás Cardozo**
* **Institución:** Pontificia Universidad Javeriana (Bogotá, Colombia)
* **Programa:** Pregrado en Ciencia de Datos (*4º Semestre*)
* **GitHub:** [@nicolascardozo0709](https://github.com/nicolascardozo0709)
* **LinkedIn:** [linkedin.com/in/nicolas-cardozo-a779b331a](https://www.linkedin.com/in/nicolas-cardozo-a779b331a)
* **Email:** `nicolas.cardozo0709@gmail.com`
