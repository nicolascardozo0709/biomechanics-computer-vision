# 🚀 Guía de Despliegue Público — Kinetix-AI

Esta guía te muestra cómo hacer que **cualquier persona, reclutador o familiar** pueda probar la plataforma **sin tener que clonar el código ni instalar Python en su computadora**.

---

## Opción 1: Despliegue Gratuito 24/7 en Hugging Face Spaces (Recomendado ⭐)

Hugging Face Spaces permite hospedar aplicaciones web y modelos de Inteligencia Artificial de forma **100% gratuita**, con 16 GB de RAM, sin necesidad de tarjeta de crédito y con un link público permanente.

### Pasos rápidos:
1. Crea una cuenta gratuita en [Hugging Face](https://huggingface.co/) (si no tienes una).
2. Haz clic en **New Space** en el menú superior o entra a [huggingface.co/new-space](https://huggingface.co/new-space).
3. Configura:
   - **Space Name:** `kinetix-biomechanics-engine`
   - **License:** `mit`
   - **Space SDK:** Selecciona **Docker** (Blank template).
   - **Visibility:** Public.
4. En tu terminal local o conectando GitHub:
   - Puedes conectar directamente tu repositorio de GitHub: `nicolascardozo0709/biomechanics-computer-vision`.
   - O hacer un push con git al remote de Hugging Face:
     ```bash
     git remote add space https://huggingface.co/spaces/TU_USUARIO/kinetix-biomechanics-engine
     git push space main
     ```
5. ¡Listo! En 2 minutos se construirá el contenedor Docker y tendrás una URL pública:
   👉 `https://huggingface.co/spaces/TU_USUARIO/kinetix-biomechanics-engine`

Cualquier persona desde su teléfono o PC puede abrir ese link, subir sus videos o hacer clic en los botones de demo.

---

## Opción 2: Compartir un Link Público Instantáneo desde tu Máquina (Túnel Temporal)

Si estás ejecutando el servidor local en tu máquina (`python app.py` o `uvicorn app:app --port 8000`), puedes generar un enlace público HTTPS seguro en 10 segundos:

### Con LocalTunnel:
Abre una terminal y ejecuta:
```bash
npx localtunnel --port 8000
```
Te dará una URL como:
```
your url is: https://cool-squat-ai.loca.lt
```
*(Nota: Al abrir ese link por primera vez, te pide ingresar tu IP pública una sola vez por seguridad).*

### Con Cloudflare Tunnel:
```bash
cloudflared tunnel --url http://127.0.0.1:8000
```
Te genera un enlace seguro directo de Cloudflare sin pantallas intermedias.

---

## Opción 3: Despliegue en Render / Railway / Fly.io

El repositorio incluye un `Dockerfile` optimizado y validado.
1. Crea una aplicación en [Render.com](https://render.com) o [Railway.app](https://railway.app).
2. Conecta tu repositorio de GitHub `biomechanics-computer-vision`.
3. Selecciona el entorno **Docker**.
4. El servicio detectará el `Dockerfile` y desplegará la API y el Dashboard automáticamente.

---

## 💡 Pruebas Rápidas con Demos Integrados

Para que los visitantes no tengan que grabar un video obligatorio para probar el sistema, el dashboard cuenta con **Demos de 1 Clic**:
- **🏋️ Probar Demo Sentadilla:** Carga y evalúa una sentadilla olímpica profunda con análisis completo de rodilla y torso.
- **💪 Probar Demo Push-Up:** Carga y evalúa flexiones de pecho estándar militar con flexión de codo y alineación de plancha.
