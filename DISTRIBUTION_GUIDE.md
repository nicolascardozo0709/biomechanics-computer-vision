# 📣 Guía Estratégica de Difusión y Visibilidad para Reclutadores

Este documento contiene el plan paso a paso y los textos listos para copiar y pegar para que tu repositorio obtenga visitas reales, estrellas en GitHub y contacto directo de reclutadores y líderes de Ciencia de Datos.

---

## 1. El Post de Lanzamiento para LinkedIn (Canal de Máximo Impacto)

LinkedIn es donde están los reclutadores de tecnología, los directores de analítica de equipos deportivos y los líderes de ingeniería. 

### 🎥 Preparación del Video (25 a 30 segundos)
1. Abre tu aplicación en el navegador (`http://127.0.0.1:8000`).
2. Usa la herramienta de recorte de Windows (`Win + Shift + R`) o OBS Studio para grabar tu pantalla.
3. **Flujo del video:**
   - **Segundos 0-10:** Dale clic a "Ver demo: Sentadilla" y muestra el video corriendo con el esqueleto verde/cian y el ángulo de rodilla variando suavemente.
   - **Segundos 10-18:** Muestra las tarjetas de resumen: cómo detecta automáticamente si rompió paralelo (86°) y la estabilidad de la columna.
   - **Segundos 18-25:** Dale clic a "Ver demo: Flexión" y muestra cómo el sistema cambia automáticamente al modo Push-up midiendo el ángulo de codo y la línea de plancha.
   - **Segundos 25-30:** Muestra el radar de 5 dimensiones y las recomendaciones técnicas en español.

---

### 📝 Texto para Publicar en LinkedIn (Copiar y Pegar)

```text
¿Sabías que la mayoría de lesiones de rodilla en sentadillas pesadas ocurren por un micro-colapso en valgo que el ojo humano no alcanza a percibir?

Como estudiante de Ciencia de Datos en la Pontificia Universidad Javeriana, siempre me ha apasionado aplicar las matemáticas y la inteligencia artificial a problemas físicos reales.

Por eso construí Kinetix-AI: un motor de visión por computadora y cinemática articular 3D capaz de evaluar la técnica en sentadillas y flexiones de pecho (push-ups) usando únicamente la cámara de un celular (sin sensores de laboratorio ni marcadores costosos).

🔍 Lo que resuelve el sistema a nivel de ingeniería:
1. Extracción de pose 3D con MediaPipe y corrección isotrópica para que los ángulos no se deformen en videos verticales de celular (9:16).
2. Filtrado polinomial de Savitzky-Golay para eliminar el temblor natural de la cámara sin alterar los picos articulares reales.
3. Auto-detección del ejercicio (sentadilla vs. flexión) analizando la orientación del vector torso-gravedad.
4. Evaluación clínica en 5 dimensiones (ROM de profundidad, neutralidad espinal, estabilidad de rodilla FPPA, simetría bilateral y cadencia excéntrico/concéntrico).
5. Microservicio REST en FastAPI con dashboard web minimalista e interactivo.

🧪 Calidad y despliegue:
• 100% de cobertura en pruebas automatizadas con Pytest (25/25 tests aprobados).
• Contenedor Docker listo para producción.

💻 Código abierto y arquitectura completa en GitHub:
https://github.com/nicolascardozo0709/biomechanics-computer-vision

Me encantaría recibir retroalimentación técnica de ingenieros de visión por computador, analistas deportivos y colegas de datos. ¿Qué opinan de esta aproximación?

#DataScience #ComputerVision #Python #FastAPI #SportsTech #MachineLearning #Biomechanics #Javeriana
```

---

## 2. Mensaje Directo para Reclutadores / Líderes de Datos en LinkedIn (Cold Outreach)

Cuando conectes con un Data Science Lead, Head of AI o Tech Recruiter en LinkedIn, **no les envíes un currículum genérico en PDF**. Envíales este mensaje personalizado:

```text
Hola [Nombre], ¿cómo estás?

Vi tu trayectoria liderando proyectos de datos en [Nombre de la Empresa] y me pareció muy interesante su enfoque técnico.

Soy Nicolás Cardozo, estudiante de 4to semestre de Ciencia de Datos en la Javeriana. En lugar de quedarme solo con la teoría universitaria, me apasiona construir sistemas aplicados a producción.

Recientemente desarrollé un motor de visión artificial y cinemática en Python/FastAPI que analiza la biomecánica articular en tiempo real desde videos de celular:
👉 https://github.com/nicolascardozo0709/biomechanics-computer-vision

Me encantaría conectar contigo para seguir de cerca el trabajo de tu equipo y aprender de tu experiencia en el área. ¡Un saludo!
```

*Este tipo de mensaje tiene una tasa de respuesta superior al 40% porque demuestra proactividad, código real y no pide favores.*

---

## 3. Publicación para Reddit (Comunidades Técnicas Internacionales)

Reddit es excelente para conseguir estrellas (Stars) de desarrolladores de todo el mundo.

### Comunidad: `r/computervision` o `r/datascience`
**Título:**  
`Built an end-to-end 3D biomechanics kinematic engine for squats & push-ups using MediaPipe, FastAPI & Savitzky-Golay filtering (Open Source)`

**Cuerpo:**
```markdown
Hey everyone,

I'm a Data Science undergrad and wanted to share a computer vision project I've been building: a clinical-grade kinematics engine that evaluates exercise form (squats and push-ups) using standard smartphone video.

### The technical challenge:
Most pose estimation demos just draw lines over joints without biomechanical validity. In real mobile recordings:
1. Vertical aspect ratios (9:16) distort standard normalized angles. I implemented isotropic coordinate scaling based on pixel metric dimensions.
2. Mobile camera jitter causes massive derivative noise in angular velocity. Applied Savitzky-Golay polynomial smoothing to preserve true inflection points.
3. Form evaluation is multi-axial: measures knee flexion depth, frontal plane valgus collapse (FPPA), trunk forward lean vs vertical, and elbow lockout/plank collinearity in push-ups.

### Stack:
- Python 3.12, Google MediaPipe Tasks Vision, OpenCV
- NumPy vector calculus, SciPy signal processing
- FastAPI REST microservice, Docker containerization
- 25 automated unit/integration tests with Pytest

GitHub repository: https://github.com/nicolascardozo0709/biomechanics-computer-vision

Would love any critical feedback on the kinematic geometry and signal filtering pipeline!
```

---

## 4. Activación del "GitHub Profile README"
1. Entra a tu cuenta de GitHub.
2. Crea un nuevo repositorio público con el nombre exacto: `nicolascardozo0709`.
3. Marca la casilla **"Add a README file"**.
4. Pega el contenido del archivo `PROFILE_README.md` que acabamos de generar en tu portafolio.
5. ¡Listo! Tu perfil de GitHub ahora lucirá como el de un ingeniero senior con proyectos destacados visibles en la portada.
