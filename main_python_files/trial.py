import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import gradio as gr
import random
from flask import Flask, request, render_template, redirect, url_for, jsonify
import sqlite3
from datetime import datetime
from flask_cors import CORS
import threading
import os 
import cv2
import json
import base64
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

import numpy as np
import matplotlib.pyplot as plt

from fuzzywuzzy import process
#import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
#import gradio as gr
# Initialize the Flask app

trial = Flask(__name__, template_folder='templates')
CORS(trial)
trial.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Load the pre-trained model and tokenizer for art review
model_name = "nlpconnect/vit-gpt2-image-captioning"
model = VisionEncoderDecoderModel.from_pretrained(model_name)
feature_extractor = ViTImageProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the BlenderBot model and tokenizer
chatbot_model_name = "facebook/blenderbot-400M-distill"
chatbot_model = BlenderbotForConditionalGeneration.from_pretrained(chatbot_model_name)
chatbot_tokenizer = BlenderbotTokenizer.from_pretrained(chatbot_model_name)

# Load the pre-trained BERT model and tokenizer
qa_model_name = "bert-base-uncased"
qa_bert_model = BertModel.from_pretrained(qa_model_name)
qa_tokenizer = BertTokenizer.from_pretrained(qa_model_name)

# Initialize conversation history for the chatbot
conversation_history = []

# Function to generate caption for an art image
def generate_caption(image):
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    output_ids = model.generate(pixel_values, max_length=16, num_beams=4)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

# Simulate AI-generated feedback based on the user description
def generate_feedback(description):
    feedback_templates = [
        "This piece demonstrates a strong {}. However, to improve, consider enhancing the {}.",
        "The use of {} in this artwork is excellent. To elevate this piece, you might want to focus on improving the {}.",
        "Great work on the {}! To take this to the next level, consider adding more {}.",
        "The {} in this artwork is well-executed. For further improvement, pay attention to the {}.",
        "Your approach to {} is commendable. For a more refined outcome, try working on the {}."
    ]

    elements = [
        "composition", "color balance", "contrast", "lighting", "texture", "depth",
        "shading", "line work", "form", "perspective", "symmetry", "proportion", "realism",
        "abstract elements", "movement", "pattern", "rhythm", "space", "scale", "structure",
        "background integration", "foreground emphasis"
    ]
    improvements = [
        "details", "dynamic range", "harmony", "focus", "perspective", "proportions",
        "brushwork", "layering", "balance", "emphasis", "unity", "variety", "repetition",
        "gradation", "transitions", "sharpness", "softness", "color palette", "edge quality",
        "visual interest", "negative space usage", "depth perception"
    ]

    feedback_parts = []
    while len(feedback_parts) < 5:
        template = random.choice(feedback_templates)
        element = random.choice(elements)
        improvement = random.choice(improvements)
        feedback_parts.append(template.format(element, improvement))
        elements.remove(element)
        improvements.remove(improvement)

    detailed_feedback = " ".join(feedback_parts)
    return detailed_feedback

# Setting up the Gradio interface for art review
def review_art(image, description):
    if not description:
        caption = generate_caption(image)
        description = caption
    feedback = generate_feedback(description)
    review = f"Art Review: {description}\n\nAI Feedback: {feedback}"
    return review

iface1 = gr.Interface(
    fn=review_art,
    inputs=[
        gr.Image(type="pil", label="Upload your art"),
        gr.Textbox(lines=4, placeholder="Describe your art...", label="Art Description")
    ],
    outputs="text",
    title="Artistic Review Tool – AI Insights for Your Masterpieces.",
    description="Upload an art image and get an AI-generated review and feedback.",
)


@trial.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@trial.route('/art_reviewer')
def art_reviewer():
    return render_template('art_reviewer.html')

@trial.route('/chatbot', methods=['POST'])
def handle_prompt():
    data = request.get_data(as_text=True)
    data = json.loads(data)
    input_text = data['prompt']
    
    # Create conversation history string
    history = "\n".join(conversation_history[-10:])  # Limit history to last 10 exchanges

    # Tokenize the input text and history
    input_ids = chatbot_tokenizer(history + "\n" + input_text, return_tensors="pt")['input_ids'][0]
    
    # Ensure the length does not exceed the maximum
    max_length = chatbot_tokenizer.model_max_length
    if len(input_ids) > max_length:
        input_ids = input_ids[-max_length:]
    
    # Create a new tensor with the truncated input_ids
    inputs = {'input_ids': input_ids.unsqueeze(0)}
    
    # Generate the response from the model
    outputs = chatbot_model.generate(inputs['input_ids'], max_length=1000, num_beams=5, early_stopping=True)

    # Decode the response
    response = chatbot_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Add interaction to conversation history
    conversation_history.append(f"User: {input_text}")
    conversation_history.append(f"Bot: {response}")

    return response



"""
@trial.route("/canvas")
def canvas():
    return render_template("canvas.html")
"""

qa_pairs = {
   "List out the drawing techniques?": "1. Line Drawing Techniques\n2. Shading Techniques\n3. Rendering Techniques\n4. Perspective Techniques\n5. Mixed Media Techniques",
    "What are the basic principles of shading?": "1. Light Source\n2. Cast Shadow\n3. Form Shadow\n4. Highlight\n5. Reflected Light",
    "Name some common perspective techniques in drawing": "1. One-Point Perspective\n2. Two-Point Perspective\n3. Three-Point Perspective\n4. Atmospheric Perspective\n5. Foreshortening",
    "give the details of Line Drawing Techniques": """Contour Drawing\n
- Definition: Contour drawing involves drawing the outline of an object or figure without lifting the drawing tool. It emphasizes the edges and contours to create a sense of form and volume.\n
- Purpose: Helps in understanding the structure and proportions of subjects. Enhances hand-eye coordination and observational skills.\n
Gesture Drawing\n
- Definition: Gesture drawing captures the essence and movement of a subject using quick, expressive lines. It focuses on capturing action, energy, and fluidity rather than details.\n
- Purpose: Improves understanding of movement and dynamics. Develops spontaneity and the ability to capture poses quickly.\n
Cross-Contour Drawing\n
- Definition: Cross-contour drawing involves drawing lines that follow the contours of the form, emphasizing its three-dimensional structure. Lines wrap around the object, indicating its volume.\n
- Purpose: Enhances understanding of form and solidity. Helps in depicting surfaces and textures realistically.\n
""",
    "give the details of Line Drawing Techniques": """Contour Drawing\n
- Definition: Contour drawing involves drawing the outline of an object or figure without lifting the drawing tool. It emphasizes the edges and contours to create a sense of form and volume.\n
- Purpose: Helps in understanding the structure and proportions of subjects. Enhances hand-eye coordination and observational skills.\n
Gesture Drawing\n
- Definition: Gesture drawing captures the essence and movement of a subject using quick, expressive lines. It focuses on capturing action, energy, and fluidity rather than details.\n
- Purpose: Improves understanding of movement and dynamics. Develops spontaneity and the ability to capture poses quickly.\n
Cross-Contour Drawing\n
- Definition: Cross-contour drawing involves drawing lines that follow the contours of the form, emphasizing its three-dimensional structure. Lines wrap around the object, indicating its volume.\n
- Purpose: Enhances understanding of form and solidity. Helps in depicting surfaces and textures realistically.\n
""",
    "give the details of Shading Techniques": """Hatching and Cross-Hatching\n
- Definition: Hatching involves drawing parallel lines to create value and shading. Cross-hatching adds intersecting lines to build up tones and textures.\n
- Purpose: Provides control over light and shadow. Creates depth and form using controlled mark-making.\n
Stippling\n
- Definition: Stippling uses dots or small marks to create value and shading. Denser clusters of dots create darker areas, while sparse dots create lighter areas.\n
- Purpose: Creates textures and gradients. Allows for precise control over tones and details.\n
Blending\n
- Definition: Blending involves smudging or gently rubbing drawing materials (e.g., graphite, charcoal) to create smooth transitions between tones.\n
- Purpose: Achieves soft gradients and realistic textures. Blurs harsh lines for a more natural look.\n
""",
    "give the details of Rendering Techniques": """Chiaroscuro\n
- Definition: Chiaroscuro is the technique of using strong contrasts between light and dark to create a sense of volume and three-dimensionality.\n
- Purpose: Emphasizes dramatic lighting effects. Enhances the illusion of depth and form.\n
Sgraffito\n
- Definition: Sgraffito involves scratching into a layer of pigment to reveal underlying layers, creating textures and adding highlights.\n
- Purpose: Adds texture and visual interest. Creates expressive and dynamic surfaces.\n
""",
    "give the details of Perspective Techniques": """One-Point Perspective\n
- Definition: One-point perspective uses a single vanishing point on the horizon line to create the illusion of depth and space in a drawing.\n
- Purpose: Provides a realistic representation of spatial relationships. Useful in architectural and interior drawings.\n
Two-Point Perspective\n
- Definition: Two-point perspective uses two vanishing points on the horizon line to depict objects and scenes from an angle, showing depth and height.\n
- Purpose: Creates dynamic compositions. Useful for drawing buildings, streets, and interiors from different viewpoints.\n
Atmospheric Perspective\n
- Definition: Atmospheric perspective simulates the effect of atmospheric conditions on the perception of distance. Objects appear lighter, cooler, and less detailed as they recede into the background.\n
- Purpose: Creates depth and realism in landscapes and outdoor scenes. Emphasizes spatial relationships and depth of field.\n
""",
    "give the details of Mixed Media Techniques": """Collage\n
- Definition: Collage involves combining different materials such as paper, fabric, photographs, and found objects to create compositions.\n
- Purpose: Allows for experimentation and creativity. Adds texture and layers to artworks.\n
Mixed Media Drawing\n
- Definition: Mixed media drawing combines various drawing materials and techniques (e.g., ink, watercolour, pastels) to create diverse effects and textures.\n
- Purpose: Expands creative possibilities. Combines the strengths of different media for expressive and layered artworks.\n
""",
    "list out the painting methods": """1.Oil Painting
2.Acrylic Painting
3.Watercolor Painting
4.Digital Painting
""",
    "give the details of Oil Painting": """Oil painting is a traditional and versatile method known for its rich colors and blending capabilities. Here’s a breakdown of the process:

1. Materials: Artists use oil paints made from pigments mixed with a binder, usually linseed oil. They paint on canvases stretched over wooden frames.
   
2. Technique:
   - Underpainting: Many artists start with an underpainting, often in monochrome, to establish values and composition.
   - Layering: Oil paints are applied in layers, allowing for blending and adjustments as the painting progresses.
   - Drying Time: Oil paints have a slow drying time, which allows for more flexibility in blending and corrections.
   
3. Tools: Brushes of various sizes and shapes are commonly used, along with palette knives for texture and detail.
""",
    "give the details of Acrylic Painting": """Acrylic painting is known for its fast drying time and versatility. Here’s how it typically works:

1. Materials: Acrylic paints are water-based and consist of pigments suspended in acrylic polymer emulsion. They can be used on various surfaces, including canvas, wood, and paper.

2. Technique:
   - Fast Drying: Acrylics dry quickly, which limits blending time but allows for layering and overpainting rapidly.
   - Opacity and Transparency: Acrylic paints can be used in thin washes like watercolors or applied thickly like oils.
   - Cleanup: Brushes and palettes are cleaned with water, making acrylics more convenient than oils in terms of cleanup.

3. Tools: Similar to oil painting, brushes and palette knives are used, along with additives like retarders to extend drying time.
""",
"give the details of Watercolor Painting": """Watercolor is known for its transparency and luminosity. Here’s an overview of the process:

1. Materials: Watercolor paints consist of pigments suspended in a water-soluble binder, usually gum arabic. They are applied to watercolor paper.

2. Technique:
   - Transparency: Watercolors are applied in washes, starting with light colors and building up to darker tones.
   - Wet-on-Wet vs. Wet-on-Dry: Techniques like wet-on-wet (applying paint to a wet surface) create soft edges, while wet-on-dry (applying paint to a dry surface) allows for more control and detail.
   - Layering: Watercolors are typically built up in layers, with each layer adding depth and complexity.

3. Tools: Brushes with soft bristles are used, along with water containers for mixing and diluting colors.
""",
"give the details of Digital Painting": """Digital painting involves creating artwork digitally using software and a graphics tablet. Here’s how it differs:

1. Materials: Artists use software like Adobe Photoshop, Corel Painter, or Procreate on a computer or tablet. A stylus and graphics tablet simulate traditional brushes and canvas.

2. Technique:
   - Layering and Blending: Similar to traditional painting, digital artists work in layers, adjusting opacity and blending modes.
   - Tools: Various digital brushes mimic traditional media, including oils, acrylics, watercolors, and specialized effects like texture brushes and smudge tools.
   - Undo and Edit: Digital painting allows for easy correction and experimentation without the constraints of traditional media.

3. Workflow: Artists can sketch directly onto the digital canvas, paint with various brushes, and utilize tools for precise editing and color adjustment.
""",
"give the details of Digital Sketching": """Digital sketching involves creating preliminary drawings or concept sketches using digital tools like tablets and software. It offers flexibility and a range of creative possibilities.

1. Tools and Software:
   - Graphics Tablets: Artists use styluses or pens on sensitive surfaces that translate pressure and movement into digital strokes.
   - Software: Popular software includes Adobe Photoshop, Procreate, Corel Painter, and SketchBook Pro, offering various brush presets and customization options.

2. Techniques:
   - Sketching: Artists begin with rough outlines and basic shapes, gradually refining details.
   - Layers: Digital sketches are often created using layers, allowing artists to work on different elements separately and adjust opacity for better control.
   - Custom Brushes: Artists can create or download custom brushes to achieve different textures and effects, enhancing the sketching process.

3. Benefits:
   - Undo and Redo: Digital sketching allows for easy corrections and experimentation without damaging the original sketch.
   - Color Adjustments: Artists can quickly test different color schemes or adjust hues and saturation levels.
   - Exportability: Sketches can be easily exported in various formats for further refinement or sharing.
""",
"give the details of Photo Manipulation": """Photo manipulation involves altering or enhancing photographs using digital tools to create artistic or surreal effects.

1. Tools and Software:
   - Adobe Photoshop: Widely used for its comprehensive tools, layers, and adjustment options.
   - GIMP: An open-source alternative offering similar functionalities to Photoshop.
   - Lightroom: Used for non-destructive editing and color correction.

2. Techniques:
   - Retouching: Correcting imperfections, enhancing details, and adjusting lighting and colors.
   - Compositing: Combining multiple images to create a new scene or concept.
   - Special Effects: Adding filters, textures, and digital painting elements to alter the photo's appearance.

3. Skills:
   - Selection Tools: Mastery of selection tools like lasso, magic wand, and pen tool for precise editing.
   - Layers and Masks: Understanding layers for non-destructive editing and masks for selective adjustments.
   - Color Grading: Enhancing mood and tone through color adjustments and gradients.

4. Ethical Considerations:
   - Authenticity: Maintaining transparency in digital alterations, especially in journalistic or documentary contexts.
   - Copyright: Respecting copyright laws when using and manipulating images.
""",
"list out the elements of art": """The elements of art are foundational concepts that artists use to create visual compositions:

1. Line: A mark made by a moving point, can be straight, curved, thick, thin, or implied.
   
2. Shape: A defined area with two dimensions (height and width), such as geometric shapes or organic forms.

3. Form: A three-dimensional object or the illusion of depth in a two-dimensional artwork.

4. Value: The lightness or darkness of tones or colors. It helps create contrasts and define shapes.

5. Texture: The surface quality of a material, can be actual (tactile) or implied (visual).

6. Color: The visual sensation produced by light as it interacts with pigments. It includes hue, saturation, and brightness.

7. Space: The area around, between, or within components of a piece. It includes positive (occupied by shapes) and negative (empty) space.

""",
"list out the principles of art": """The principles of art guide how artists organize the elements of art within their work:

1. Balance: The distribution of visual weight in a composition. It can be symmetrical, asymmetrical, or radial.

2. Emphasis: The focal point of an artwork that draws attention and creates dominance.

3. Movement: The path the viewer's eye takes through a composition, guided by elements like lines, shapes, and contrast.

4. Pattern: Repetition of elements like shapes, colors, or textures to create visual interest.

5. Rhythm: The repetition of elements to create a sense of movement or flow.

6. Proportion: The comparative relationship of one part to another in size, quantity, or degree.

7. Unity: The harmonious relationship among elements and principles in a composition.

""",
"list out the art movements": """Art movements are styles or tendencies in art that have specific common goals, techniques, or themes:

1. Renaissance: A period of cultural and artistic rebirth in Europe, characterized by realistic representation, perspective, and humanism.

2. Impressionism: 19th-century movement focused on capturing the impression of a scene using light and color, often with visible brushstrokes.

3. Surrealism: A 20th-century movement that explored the unconscious mind, dreams, and fantastic imagery.

4. Abstract Expressionism: Post-World War II movement emphasizing spontaneous, automatic, or subconscious creation, often with large-scale canvases and gestural marks.

5. Pop Art: A 20th-century movement that used popular culture and mass media as its subject matter, often employing irony or parody.

""",
"list out the  art techniques": """ Techniques refer to the methods and processes artists use to create their artworks:

1. Drawing: The use of lines and shading to create shapes and forms.

2. Painting: Application of paint to surfaces using brushes, palette knives, or other tools.

3. Sculpture: Three-dimensional art forms created by modeling, carving, or assembling materials like clay, stone, metal, or wood.

4. Printmaking: Creating artworks by transferring ink from a matrix or plate to paper or another surface.

5. Photography: Capturing images using light-sensitive materials, often involving composition and manipulation of light.

6. Digital Art: Creation of art using digital tools and techniques, such as digital painting, 3D modeling, and photo manipulation.

""",
"step by step tutorial for oil painting": """  Step-by-Step Tutorial for Oil Painting 

Materials Needed:
- Oil paints
- Brushes (variety of sizes)
- Palette and palette knife
- Canvas
- Easel
- Linseed oil or medium
- Mineral spirits or turpentine
- Rags/paper towels
- Apron/old clothes
- Pencil/charcoal

Steps:

1. Prepare Workspace:
   - Set up easel, organize materials, and ensure good lighting.

2. Prepare Canvas:
   - Apply gesso if needed and sketch composition.

3. Underpainting:
   - Apply a thin wash of one color (e.g., burnt umber) to tone the canvas and sketch basic shapes.

4. Blocking in Colors:
   - Use large brushes to block in main colors and values.

5. Establish Values and Forms:
   - Build up layers, blend colors, and focus on light and shadow.

6. Adding Details:
   - Use smaller brushes for details, highlights, and dark accents.

7. Glazing (Optional):
   - Apply thin, transparent layers of color over dry areas for depth.

8. Final Touches:
   - Make adjustments, add final details, and let the painting dry completely.

9. Drying and Varnishing:
   - Allow painting to dry thoroughly, then apply varnish.

Tips:
- Work "fat over lean": Start with thinner layers, then thicker ones.
- Clean brushes with mineral spirits and soap.
- Be patient and allow layers to dry to avoid muddiness.
- Experiment with techniques and textures.

""",
"step by step tutorial for watercolor painting": """Step-by-Step Tutorial for Watercolor Painting 

Materials Needed:
- Watercolor paints
- Brushes (variety of sizes, including round and flat)
- Watercolor paper
- Palette
- Water containers
- Paper towels or rags
- Pencil (for sketching)
- Masking tape (optional)

Steps:

1. Prepare Workspace:
   - Set up your materials in a well-lit area, and secure your watercolor paper to a board or table using masking tape.

2. Prepare Paper:
   - Lightly sketch your composition on the watercolor paper with a pencil.

3. Mix Colors:
   - Use the palette to mix the colors you’ll need, starting with light washes and gradually working towards more intense colors.

4. Wet-on-Wet Technique:
   - Wet the paper where you want soft, blended colors, then apply watercolor paint. This creates smooth transitions and effects.

5. Wet-on-Dry Technique:
   - Apply watercolor paint directly onto dry paper for more control and sharper edges.

6. Layering and Building Up Color:
   - Start with light washes and gradually build up layers to create depth and dimension. Allow each layer to dry before applying the next.

7. Adding Details:
   - Use smaller brushes to add fine details and final touches once the base layers are dry.

8. Creating Texture:
   - Experiment with techniques like splattering, lifting paint with a dry brush, or using salt to create interesting textures.

9. Final Touches:
   - Make any necessary adjustments to color and detail, and remove any masking tape carefully once the painting is dry.

10. Drying and Finishing:
    - Let the painting dry completely, and if desired, flatten the paper by placing it under a heavy book or using a gentle iron on the back.

Tips:
- Use clean water to avoid muddy colors.
- Work from light to dark, as it’s easier to add darker colors than to lighten areas.
- Experiment with different techniques to find your style.
- Keep your brushes and palette clean.

""",
"step by step tutorial for acrylic painting": """ Step-by-Step Tutorial for Acrylic Painting 

Materials Needed:
- Acrylic paints
- Brushes (variety of sizes and shapes)
- Palette
- Canvas or canvas board
- Easel
- Water container
- Paper towels or rags
- Palette knife (optional)
- Acrylic medium (e.g., matte or gloss medium)
- Pencil (for sketching)
- Apron or old clothes

Steps:

1. Prepare Workspace:
   - Set up your easel in a well-lit area and organize your materials.

2. Prepare Canvas:
   - Lightly sketch your composition on the canvas with a pencil.

3. Mix Colors:
   - Use the palette to mix your colors, starting with basic hues and blending to create the desired shades.

4. Blocking in Colors:
   - Use large brushes to block in the main shapes and colors, focusing on the overall composition.

5. Building Layers:
   - Gradually build up layers of paint, allowing each layer to dry before adding the next. Acrylics dry quickly, making it easy to layer.

6. Adding Details:
   - Use smaller brushes to add fine details and refine shapes. Work from general to specific, adding detail as you go.

7. Blending and Texture:
   - Use a palette knife or different brush techniques to create texture and blend colors smoothly.

8. Highlighting and Shadows:
   - Add highlights and shadows to create depth and dimension. Acrylics can be layered to adjust the intensity of light and dark areas.

9. Final Touches:
   - Make any final adjustments to color, detail, and composition. Add any finishing touches to enhance the painting.

10. Sealing the Painting:
    - Once the painting is completely dry, apply a layer of acrylic varnish to protect the surface and enhance the colors.

Tips:
- Work quickly, as acrylics dry fast. You can use a slow-drying medium to extend working time.
- Clean brushes immediately after use to prevent paint from drying on them.
- Experiment with various techniques like dry brushing, glazing, and scumbling.
- Keep a spray bottle of water handy to keep your palette moist.
""",
"step by step tutorial for digital painting": """ Step-by-Step Tutorial for Digital Painting 

Materials Needed:
- Digital painting software (e.g., Adobe Photoshop, Procreate, Krita)
- Graphics tablet and stylus
- Computer or tablet
- Reference images (optional)

Steps:

1. Prepare Workspace:
   - Set up your computer or tablet and ensure your graphics tablet is connected and calibrated.

2. Open Digital Painting Software:
   - Launch your chosen digital painting software and create a new canvas with your desired dimensions and resolution.

3. Sketch the Composition:
   - Use a basic brush to sketch your composition on a new layer. This initial sketch helps plan the overall layout.

4. Blocking in Colors:
   - Create new layers for different parts of your painting (e.g., background, foreground). Block in the main shapes and colors using a larger brush.

5. Building Layers:
   - Add new layers for more details and refinements. Work from the background to the foreground, gradually adding complexity to the painting.

6. Adding Details:
   - Use smaller brushes and different textures to add fine details. Take advantage of the software's brush settings to achieve various effects.

7. Blending and Shading:
   - Use blending tools or brushes to smooth transitions between colors and create realistic shading. Adjust opacity and flow settings for subtle effects.

8. Highlighting and Shadows:
   - Add highlights and shadows to create depth and dimension. Use layer modes like Multiply for shadows and Overlay for highlights.

9. Texture and Effects:
   - Experiment with different brushes and textures to add depth and interest. You can use custom brushes or textures available online.

10. Final Touches:
    - Zoom out to evaluate the overall composition and make any final adjustments to colors, details, and layers.

11. Save and Export:
    - Save your work regularly in the software's native format to preserve layers. Export the final image in your desired format (e.g., JPEG, PNG).

Tips:
- Use keyboard shortcuts to speed up your workflow.
- Utilize layers to separate different elements and make adjustments easier.
- Experiment with different brushes and settings to find your style.
- Use reference images to guide proportions, lighting, and color schemes.

""",
"step by step tutorial for digital sketching": """  Step-by-Step Tutorial for Digital Sketching 

Materials Needed:
- Digital sketching software (e.g., Adobe Photoshop, Procreate, Krita)
- Graphics tablet and stylus
- Computer or tablet

Steps:

1. Prepare Workspace:
   - Set up your computer or tablet and connect your graphics tablet.

2. Open Digital Sketching Software:
   - Launch your chosen sketching software and create a new canvas with your desired dimensions and resolution.

3. Choose a Brush:
   - Select a basic brush for sketching. Adjust the brush size and opacity to your preference.

4. Sketch the Basic Shapes:
   - On a new layer, lightly sketch the basic shapes of your composition. Focus on proportions and overall layout.

5. Refine the Sketch:
   - Create a new layer and refine your sketch, adding more detail and clarity. Use the initial sketch as a guide.

6. Add Details:
   - Continue refining your sketch on additional layers if needed. Add finer details and textures.

7. Adjust and Edit:
   - Use the software’s tools to adjust lines, proportions, and details. Utilize tools like Transform, Eraser, and Selection.

8. Finalize the Sketch:
   - Merge layers if necessary and make final adjustments. Clean up any stray lines or marks.

9. Save and Export:
   - Save your sketch in the software’s native format to preserve layers. Export the final sketch in your desired format (e.g., JPEG, PNG).

Tips:
- Use reference images to guide your proportions and details.
- Utilize keyboard shortcuts for efficiency.
- Experiment with different brushes and settings to find your style.
- Save your work regularly to avoid losing progress.

""",
"step by step tutorial for photo manipulation": """Step-by-Step Tutorial for Photo Manipulation 

Materials Needed:
- Photo manipulation software (e.g., Adobe Photoshop, GIMP, Affinity Photo)
- High-resolution photos to work with
- Graphics tablet (optional)

Steps:

1. Prepare Workspace:
   - Set up your computer and open your photo manipulation software.

2. Import Photos:
   - Import the photos you’ll be working with into the software.

3. Basic Adjustments:
   - Use tools like Crop, Rotate, and Resize to prepare your images. Adjust brightness, contrast, and levels as needed.

4. Selection and Masking:
   - Use selection tools (e.g., Lasso, Magic Wand) to isolate parts of your images. Create masks to hide or reveal parts of the images.

5. Combine Images:
   - Use layers to combine different images. Adjust the layer opacity and blending modes to create seamless compositions.

6. Retouching and Healing:
   - Use tools like Clone Stamp, Healing Brush, and Spot Healing to remove imperfections and blend edges.

7. Color Correction:
   - Adjust color balance, saturation, and hue to ensure consistency across the composition.

8. Adding Effects:
   - Apply filters and effects to enhance the image. Experiment with options like Blur, Sharpen, and Noise.

9. Final Touches:
   - Add any final adjustments, such as text, borders, or additional elements. Ensure all elements blend well together.

10. Save and Export:
    - Save your work in the software’s native format to preserve layers. Export the final image in your desired format (e.g., JPEG, PNG).

Tips:
- Use non-destructive editing techniques like adjustment layers and masks.
- Work with high-resolution images to maintain quality.
- Experiment with different blending modes and opacity settings.
- Keep your layers organized and labeled.
""",
"step by step tutorial for Line Drawing Techniques": """Step-by-Step Tutorial for Line Drawing Techniques

Materials Needed:
- Drawing paper
- Pencils (variety of hardness, e.g., HB, 2B, 4B)
- Erasers
- Ruler (for perspective techniques)
- Drawing board or flat surface

Steps:

1. Contour Drawing:
   - Prepare Paper: Tape down paper on a flat surface to prevent movement.
   - Observe Subject: Choose a simple object and carefully observe its outlines.
   - Draw Contours: Without lifting the pencil, draw the object's outline in a single, continuous line.
   - Refine Lines: Go over the contours to refine and correct any mistakes.

2. Gesture Drawing:
   - Select Subject: Choose a dynamic subject, like a person or animal in motion.
   - Quick Sketches: Use loose, quick strokes to capture the essence of the movement.
   - Focus on Action: Emphasize the action and flow rather than details.

3. Cross-Contour Drawing:
   - Choose Subject: Select a simple 3D object, like a sphere or cylinder.
   - Draw Contours: Start with basic contour lines to outline the shape.
   - Add Cross-Contours: Draw lines across the form, wrapping around the object to suggest volume.

Tips:
- Practice regularly to improve hand-eye coordination.
- Focus on observing the subject carefully before drawing.
- Use light pressure for initial sketches to easily erase mistakes.

""",
"step by step tutorial for photo manipulation": """Step-by-Step Tutorial for Shading Techniques (Short Version)

Materials Needed:
- Drawing paper
- Graphite pencils (variety of hardness, e.g., HB, 2B, 4B)
- Blending stumps or tissue
- Erasers

Steps:

1. Hatching:
   - Draw Lightly: Begin with light, parallel lines.
   - Layer Lines: Add more lines to increase darkness and build texture.
   - Vary Pressure: Vary pencil pressure to create different shades.

2. Cross-Hatching:
   - Draw Hatch Lines: Start with basic hatching lines.
   - Add Cross-Hatch: Draw a second set of lines intersecting the first set.
   - Build Up Layers: Add more layers for deeper shadows and texture.

3. Stippling:
   - Dot Placement: Use small dots to create value.
   - Vary Density: Increase the density of dots to darken areas.
   - Build Gradually: Slowly build up values for smooth transitions.

4. Blending:
   - Apply Graphite: Use pencils to apply graphite to the paper.
   - Blend with Stump: Use blending stumps or tissue to smooth out transitions.
   - Adjust Values: Add more graphite and blend as needed for desired effect.

Tips:
- Practice creating gradients to master control over shading.
- Use different pencil hardness to achieve various textures.
- Clean blending stumps regularly to avoid muddy shading.

""",
"step by step tutorial for rendering techniques": """Step-by-Step Tutorial for Rendering Techniques 

Materials Needed:
- Drawing paper or canvas
- Pencils, charcoal, or paint (depending on medium)
- Erasers
- Blending tools

Steps:

1. Chiaroscuro:
   - Block in Shadows: Start with the darkest areas first.
   - Layer Gradually: Build up lighter values slowly, maintaining strong contrasts.
   - Blend Smoothly: Use blending tools to smooth transitions between light and dark.

2. Sgraffito:
   - Apply Base Layer: Apply a base layer of dark color.
   - Scratch Surface: Use a sharp tool to scratch into the layer, revealing lighter colors underneath.
   - Add Texture: Create textures and patterns by varying pressure and direction.

Tips:
- Experiment with different tools for scratching in sgraffito.
- Use a combination of soft and hard edges to enhance realism in chiaroscuro.
- Focus on light source and shadow consistency.
""",
"step by step tutorial for perspective techniques": """Step-by-Step Tutorial for Perspective Techniques

Materials Needed:
- Drawing paper
- Pencils
- Ruler
- Erasers

Steps:

1. One-Point Perspective:
   - Draw Horizon Line: Draw a horizontal line across the paper.
   - Add Vanishing Point: Place a single point on the horizon line.
   - Draw Guidelines: Draw lines from the vanishing point to the edges of the paper.
   - Sketch Objects: Draw objects along these guidelines to create depth.

2. Two-Point Perspective:
   - Draw Horizon Line: Draw a horizontal line across the paper.
   - Add Two Vanishing Points: Place two points on the horizon line.
   - Draw Guidelines: Draw lines from each vanishing point.
   - Sketch Objects: Draw objects using these guidelines for more complex depth.

3. Atmospheric Perspective:
   - Choose Subject: Select a landscape or outdoor scene.
   - Draw Foreground: Draw objects in the foreground with darker, more detailed lines.
   - Draw Background: Draw distant objects lighter and less detailed.
   - Add Gradients: Use lighter shades for distant elements to simulate atmospheric effect.

Tips:
- Practice drawing simple shapes in perspective before moving to complex scenes.
- Use light lines for guidelines to avoid overpowering the final drawing.
- Observe real-life scenes to understand how perspective works naturally.

""",
"step by step tutorial for mixed media techniques": """Step-by-Step Tutorial for Mixed Media Techniques 

Materials Needed:
- Variety of art materials (e.g., watercolors, ink, acrylics, collage materials)
- Brushes and pens
- Glue
- Scissors
- Drawing paper or canvas

Steps:

1. Collage:
   - Gather Materials: Collect various papers, fabrics, and found objects.
   - Plan Composition: Arrange the materials on the paper without gluing.
   - Glue Elements: Start gluing pieces in place, working from background to foreground.
   - Add Details: Use pens, inks, or paints to add final details and unify the composition.

2. Mixed Media Drawing:
   - Layer Media: Start with a base layer of watercolor or ink wash.
   - Add Drawing: Use pens or pencils to draw over the base layer.
   - Build Up Layers: Continue adding layers of different media, like acrylics or pastels.
   - Experiment: Try different techniques, like scratching into wet paint or adding collage elements.

Tips:
- Experiment with different combinations of materials to find what works best.
- Allow layers to dry completely before adding new ones.
- Use fixatives to protect delicate materials like pastels or charcoal.

""",
"give checklist for mixed media techniques": """Checklist for Mixed Media Techniques
1. Materials Prepared:
   - Variety of art materials (e.g., watercolors, ink, acrylics, collage materials)
   - Brushes and pens
   - Glue
   - Scissors
   - Drawing paper or canvas

2. Collage:
   - Materials gathered
   - Composition planned
   - Pieces glued in place from background to foreground
   - Details added with pens, inks, or paints

3. Mixed Media Drawing:
   - Base layer of watercolor or ink wash applied
   - Drawing added with pens or pencils
   - Layers of different media built up
   - Experimentation with techniques (e.g., scratching into wet paint, adding collage elements)

""",
"give checklist for perspective techniques": """Checklist for Perspective Techniques
1. Materials Prepared:
   - Drawing paper
   - Pencils
   - Ruler
   - Erasers

2. One-Point Perspective:
   - Horizon line drawn
   - Single vanishing point placed
   - Guidelines drawn from vanishing point
   - Objects sketched along guidelines

3. Two-Point Perspective:
   - Horizon line drawn
   - Two vanishing points placed
   - Guidelines drawn from each point
   - Objects sketched using guidelines

4. Atmospheric Perspective:
   - Landscape or outdoor scene chosen
   - Foreground drawn with darker, detailed lines
   - Background drawn lighter and less detailed
   - Gradients added to simulate atmospheric effect
""",
"give checklist for shading techniques": """Checklist for Shading Techniques
1. Materials Prepared:
   - Drawing paper
   - Graphite pencils (variety of hardness)
   - Blending stumps or tissue
   - Erasers

2. Hatching:
   - Light, parallel lines drawn
   - Lines layered for texture
   - Pencil pressure varied for shading

3. Cross-Hatching:
   - Basic hatching lines drawn
   - Second set of intersecting lines added
   - Layers built for deeper shadows and texture

4. Stippling:
   - Dots placed to create value
   - Dot density varied for shading
   - Gradients built gradually

5. Blending:
   - Graphite applied to paper
   - Blending tools used for smooth transitions
   - Values adjusted as needed
""",
"give checklist for rendering techniques": """Checklist for Rendering Techniques
1. Materials Prepared:
   - Drawing paper or canvas
   - Pencils, charcoal, or paint (depending on medium)
   - Erasers
   - Blending tools

2. Chiaroscuro:
   - Shadows blocked in first
   - Lighter values built gradually
   - Transitions blended smoothly

3. Sgraffito:
   - Base layer of dark color applied
   - Surface scratched to reveal lighter layers
   - Texture and patterns varied
""",
"give checklist for line drawing techniques": """Checklist for Line Drawing Techniques
1. *Materials Prepared*:
   - Drawing paper
   - Pencils (variety of hardness)
   - Erasers
   - Ruler (for perspective techniques)
   - Drawing board or flat surface

2. *Contour Drawing*:
   - Paper secured to drawing board
   - Subject chosen and observed
   - Outline drawn in a single, continuous line
   - Contours refined and corrected

3. *Gesture Drawing*:
   - Dynamic subject selected
   - Quick, loose sketches capturing movement
   - Emphasis on action and flow over detail

4. *Cross-Contour Drawing*:
   - Simple 3D object chosen
   - Basic contour lines drawn
   - Cross-contour lines added to suggest volume
""",
"give checklist for oil painting": """ Checklist for Oil Painting
1. Materials Prepared:
   - Oil paints (variety of colors)
   - Painting brushes (different sizes and shapes)
   - Palette knives
   - Canvas or oil painting paper
   - Easel
   - Linseed oil or medium
   - Palette
   - Turpentine or mineral spirits
   - Rags or paper towels

2. Preparation:
   - Canvas or painting surface prepared (primed if necessary)
   - Workspace organized with adequate ventilation
   - Palette set up with colors squeezed out and ready to use
   - Brushes and tools cleaned and prepared

3. Sketch and Composition:
   - Reference image or sketch for composition ready
   - Underdrawing lightly sketched on canvas if needed
   - Composition balanced and pleasing

4. Painting Process:
   - Blocking In:
     - Main shapes and colors blocked in using large brushes.
     - Background and foreground elements defined.
   - Layering and Blending:
     - Colors layered gradually to build depth and richness.
     - Use of medium for thinning or glazing layers.
     - Blending edges and transitions between colors.
   - Detailing:
     - Fine details added using smaller brushes.
     - Highlights and shadows enhanced to create depth.
     - Texture and brushwork varied for visual interest.

5. Finishing Touches:
   - Final adjustments made to colors and values.
   - Painting allowed to dry thoroughly before varnishing (if desired).

6. Clean-Up and Maintenance:
   - Brushes and tools cleaned with turpentine or mineral spirits.
   - Palette and mixing surfaces cleaned thoroughly.
   - Proper disposal or storage of hazardous materials (e.g., used solvents).

""",
"give checklist for acrylic painting": """Checklist for Acrylic Painting
1. Materials Prepared:
   - Acrylic paints (variety of colors)
   - Painting brushes (different sizes and shapes)
   - Palette knives
   - Canvas or acrylic painting paper
   - Easel
   - Acrylic medium or water
   - Palette
   - Water container
   - Rags or paper towels

2. Preparation:
   - Canvas or painting surface prepared (primed if necessary)
   - Workspace organized with good lighting
   - Palette set up with colors squeezed out and ready to use
   - Brushes and tools cleaned and prepared

3. Sketch and Composition:
   - Reference image or sketch for composition ready
   - Underdrawing lightly sketched on canvas if needed
   - Composition balanced and planned

4. Painting Process:
   - Blocking In:
     - Main shapes and colors blocked in using large brushes.
     - Background and foreground defined.
   - Layering and Mixing:
     - Colors layered to build depth and vibrancy.
     - Use of water or medium for thinning and blending.
     - Experimentation with different acrylic techniques (e.g., impasto, glazing).
   - Detailing:
     - Fine details added using smaller brushes or tools.
     - Highlights and shadows emphasized for dimension.
     - Texture and brushwork varied for visual appeal.

5. Finishing Touches:
   - Final adjustments made to colors and values.
   - Painting allowed to dry completely.
   - Varnishing applied for protection and enhancement (if desired).

6. Clean-Up and Maintenance:
   - Brushes and tools cleaned with water and mild soap.
   - Palette and mixing surfaces cleaned thoroughly.
   - Proper storage of acrylic paints and materials to prevent drying.

""",
"give checklist for watercolor painting": """Checklist for Watercolor Painting
1. Materials Prepared:
   - Watercolor paints (variety of colors)
   - Watercolor brushes (different sizes and shapes)
   - Watercolor paper (cold-pressed or hot-pressed)
   - Palette
   - Water container
   - Masking fluid (optional)
   - Paper towels or sponge

2. Preparation:
   - Watercolor paper stretched or taped down if necessary
   - Workspace organized with good natural or artificial lighting
   - Palette set up with colors mixed and ready
   - Brushes cleaned and prepared

3. Sketch and Composition:
   - Reference image or sketch for composition ready
   - Light pencil sketch or use of masking fluid for preserving highlights
   - Composition planned with consideration of negative space

4. Painting Process:
   - Wet-on-Wet Technique:
     - Paper wetted with clean water before applying paint.
     - Colors dropped or painted into wet areas for soft blends.
   - Layering and Glazing:
     - Gradual building of layers for depth and transparency.
     - Use of glazing techniques for richer colors and effects.
   - Detailing and Dry Brush:
     - Fine details added using dry brush techniques.
     - Textures created by lifting color or scratching into dry paint.

5. Finishing Touches:
   - Final adjustments made to colors and values.
   - Painting allowed to dry naturally or with minimal heat.
   - Use of fixatives or framing for preservation and presentation.

6. Clean-Up and Maintenance:
   - Brushes cleaned thoroughly with water and mild soap.
   - Palette and water containers cleaned promptly after use.
   - Proper storage of watercolor paints and papers to prevent damage.

""",
"give checklist for digital painting": """Checklist for Digital Painting
1. Materials Prepared:
   - Digital painting software (e.g., Adobe Photoshop, Procreate, Krita)
   - Graphics tablet and stylus
   - Computer or tablet with adequate specifications
   - Reference images or sketches (if applicable)

2. Setup and Preparation:
   - Software installed and updated to the latest version
   - Graphics tablet calibrated and functional
   - Workspace organized with good lighting and ergonomic setup

3. Sketch and Composition:
   - Reference image or initial sketch imported into software
   - Basic canvas settings configured (dimensions, resolution)
   - Composition planned and adjusted using digital tools

4. Painting Process:
   - Blocking In:
     - Main shapes and colors blocked in using larger brushes or tools.
     - Background and foreground elements established.
   - Layering and Blending:
     - Colors layered gradually to build depth and texture.
     - Use of blending modes and tools for smooth transitions.
   - Detailing and Effects:
     - Fine details added using smaller brushes or custom brushes.
     - Special effects applied (e.g., textures, lighting effects) as needed.

5. Final Adjustments and Export:
   - Final adjustments made to colors, values, and composition.
   - Painting allowed to rest or reviewed before final export.
   - Exported in preferred file format (e.g., JPEG, PNG) with appropriate settings.

6. Clean-Up and Maintenance:
   - Software preferences and custom brushes saved for future use.
   - Graphics tablet and stylus cleaned and maintained regularly.
   - Backup and storage of digital painting files for archival purposes.

""","Name some style adaptations of art": "These are some style adaptations of art\n1. Realism\n2. Impressionism\n3. Abstract Art\n4. Surrealism\n5. Cubism\n6. Expressionism",
    "What is realism": """Realism aims to depict subjects with a high degree of accuracy, often resembling a photograph. Artists focus on detail, light, and perspective to create lifelike representations.
""",
    "step by step tutorial for realism": """Step-by-Step Tutorial for Realism
1. Subject Selection: Choose a subject with clear details and realistic textures.
2. Sketching: Start with a light sketch to outline proportions and key details.
3. Value and Shading: Use gradual shading and blending techniques to achieve smooth transitions between light and dark areas.
4. Detailing: Add intricate details using fine brushes or pencils.
5. Final Touches: Evaluate proportions and refine details to enhance realism.
""",
    "give a checklist for realism": """Checklist:
- Understand proportions and anatomy.
- Master shading techniques (cross-hatching, blending).
- Focus on light and shadow interplay.
- Use high-quality references for accuracy.
- Aim for precise details in textures and surfaces.

""",
    "sample exercises for realism": """Sample Exercises:
Draw a still life setup using a single light source to practice shading and rendering textures realistically,
Draw a still life composition with attention to light and shadow,
Create a portrait with realistic details and textures.
""",
    "example for realism": """Example Artists:
- Diego Velázquez: Known for his realistic portraits and scenes.
- Chuck Close: Modern artist famous for hyper-realistic portraits.
""",

    "What is Impressionism": """Impressionism emphasizes capturing the essence of a scene through light, color, and brushwork rather than precise detail. Artists often use short, visible brushstrokes and vibrant colors to convey atmosphere and mood.
""",
    "step by step tutorial for Impressionism": """Step-by-Step Tutorial for Impressionism
1. Color Palette: Select bright and contrasting colors for your palette.
2. Loose Sketching: Begin with loose, gestural sketches to capture basic shapes and composition.
3. Brushwork: Use short, quick brushstrokes to apply colors, leaving some areas unpainted for light to shine through.
4. Light and Atmosphere: Focus on capturing the effects of light and weather conditions.
5. Final Details: Add final touches sparingly to maintain the impressionistic style.
""",
    "give a checklist for Impressionism": """Checklist:
- Use a varied color palette with emphasis on primary and complementary colors.
- Experiment with different brushstroke techniques (dabbing, stippling).
- Capture changing light conditions and atmospheric effects.
- Emphasize the overall mood and feeling of the scene.
- Avoid excessive detail and aim for spontaneity in brushwork.

""",
    "sample exercises for Impressionism": """Sample Exercises:
Paint a landscape at different times of the day to observe and interpret changing light effects,
Paint a landscape using loose brushstrokes and vibrant colors,
Capture a scene with emphasis on the overall impression rather than details.
""",
    "example for Impressionism": """Example Artists:
- Claude Monet: Famous for his series of water lilies and landscapes.
- Pierre-Auguste Renoir: Known for his portraits and lively scenes of everyday life.
""",

    "What is abstract art": """Abstract art focuses on shapes, colors, and forms rather than realistic depiction. It can be non-representational or based on real subjects but interpreted in a non-traditional way.
""",
    "step by step tutorial for abstract art": """Step-by-Step Tutorial for abstract art
1. Concept Development: Decide on the idea or emotion you want to convey abstractly.
2. Color and Composition: Choose a color scheme and experiment with different compositions.
3. Techniques: Explore various techniques such as dripping, splattering, or geometric forms.
4. Layering: Build layers of paint or other materials to create depth and texture.
5. Balance and Harmony: Ensure the artwork has visual balance and cohesive elements.

""",
    "give a checklist for abstract art": """Checklist:
- Experiment with unconventional materials and tools.
- Focus on shapes, lines, and forms rather than representation.
- Emphasize expression and personal interpretation.
- Use contrasting textures and surfaces.
- Aim for a balanced composition and unity of elements.
""",
    "sample exercises for abstract art": """Sample Exercises:
Create an abstract piece inspired by music, using rhythm and movement as your guiding principles,
Create a non-representational artwork using geometric shapes and lines,
Experiment with colors and forms to express emotions and concepts.
""",
    "example for abstract art": """Example Artists:
- Jackson Pollock: Known for his drip painting technique.
- Wassily Kandinsky: Pioneer of abstract art, focused on color and form.
""",

    "What is Surrealism": """Surrealism seeks to express the subconscious mind and dreams through bizarre and illogical scenes. Artists often combine realistic elements with fantastical or dreamlike imagery.
""",
    "step by step tutorial for Surrealism": """Step-by-Step Tutorial for Surrealism
1. Imagination and Concept: Develop a surreal concept or theme.
2. Sketching: Create a preliminary sketch to outline the composition.
3. Symbolism and Metaphor: Use symbolic elements to convey deeper meanings.
4. Distortion and Transformation: Distort proportions and merge unrelated objects.
5. Detailing: Add intricate details to enhance the surreal effect.

""",
    "give a checklist for Surrealism": """Checklist:
- Explore dreams and subconscious themes.
- Experiment with juxtaposition and unexpected combinations.
- Use symbolism and metaphor to add layers of meaning.
- Focus on creating a sense of mystery and wonder.
- Maintain a balance between realism and fantasy elements.

""",
    "sample exercises for Surrealism": """Sample Exercise:
Create a surreal self-portrait that represents your inner thoughts and emotions.

""",
    "example for Surrealism": """Example Artists:
- Salvador Dalí: Known for his melting clocks and bizarre landscapes.
- René Magritte: Famous for his bowler-hatted men and use of visual puns.

""",

    "What is Cubism": """Cubism focuses on depicting subjects from multiple viewpoints and breaking them down into geometric shapes. Artists emphasize abstract forms and the interplay of space and light.
""",
    "step by step tutorial for Cubism": """Step-by-Step Tutorial for Cubism
1. Subject Selection: Choose a subject with clear, defined shapes.
2. Geometric Abstraction: Break down the subject into geometric forms (cubes, cones, cylinders).
3. Multiple Viewpoints: Represent different angles and perspectives simultaneously.
4. Color and Texture: Experiment with muted colors and textured surfaces.
5. Collage and Assemblage: Incorporate mixed media elements for added depth.
""",
    "give a checklist for Cubism": """Checklist:
- Analyze objects from multiple viewpoints.
- Experiment with fragmentation and reassembly of forms.
- Use subdued color palettes to emphasize form over color.
- Incorporate textural elements to enhance visual interest.
- Focus on creating dynamic compositions through intersecting planes.
""",
    "sample exercises for Cubism": """Sample Exercise:
Create a cubist still life using everyday objects, exploring fragmentation and spatial relationships.
""",
    "example for Cubism": """Example Artists:
- Pablo Picasso: Co-founder of Cubism, known for his fragmented portraits and still lifes.
- Georges Braque: Known for his analytical cubist approach and use of collages.
""",

    "What is Expressionism": """Expressionism emphasizes the emotional experience of the artist, often using bold colors, exaggerated forms, and dynamic compositions to convey feelings and moods.
""",
    "step by step tutorial for Expressionism": """Step-by-Step Tutorial for Expressionism
1. Emotional Theme: Choose a theme or subject that evokes strong emotions.
2. Color Palette: Select bold and expressive colors to convey mood.
3. Gestural Brushwork: Use energetic and spontaneous brushstrokes.
4. Distortion and Symbolism: Distort forms and use symbolic elements.
5. Personal Expression: Focus on conveying your inner feelings and reactions.
""",
    "give a checklist for Expressionism": """Checklist:
- Emphasize emotional intensity and personal expression.
- Use exaggerated forms and distorted perspectives.
- Experiment with non-naturalistic colors and contrasts.
- Incorporate symbolic elements to enhance meaning.
- Aim for a sense of spontaneity and immediacy in your artwork.

""",
    "sample exercises for Expressionism": """Sample Exercise:
Paint a series of portraits that express different emotions using vivid colors and expressive brushwork.
""",
    "example for Expressionism": """Example Artists:
- Edvard Munch: Known for "The Scream" and his exploration of anxiety and existential themes.
- Egon Schiele: Expressionist known for his provocative portraits and raw emotion.
""",

    "What are the benefits of adapting different art styles?": """Adapting different art styles can enhance an artist’s versatility, creativity, and technical skills. It allows artists to express a broader range of emotions and ideas, experiment with different techniques, and understand the historical and cultural contexts of various artistic movements. This adaptability can also lead to unique personal styles that blend elements from multiple influences.
""",

    "How can an artist transition from realism to impressionism?": """To transition from realism to impressionism:
1. Simplify Detail: Focus less on intricate details and more on capturing the overall impression of a scene.
2. Color Palette: Use brighter, more vibrant colors.
3. Brushwork: Practice using quick, visible brushstrokes to depict light and movement.
4. Outdoor Painting: Paint en plein air (outdoors) to observe natural light and atmospheric conditions directly.
5. Experiment: Allow for spontaneity and experiment with different brushstroke techniques.
""",
    "What challenges might an artist face when trying to adapt to abstract art from a realistic style?": """ Challenges include:
1. Letting Go of Detail: Moving away from detailed representation to embrace abstract forms can be difficult.
2. Conceptual Thinking: Developing abstract concepts and ideas might require a different mindset.
3. Technique Adjustment: Abstract art often involves different techniques and materials, requiring new skills.
4. Viewer Interpretation: Abstract art is open to interpretation, and artists might find it challenging to convey their intended message.
5. Acceptance: Abstract art may not always be immediately understood or appreciated by audiences accustomed to realism.
""",
    "How does understanding the historical context of an art style aid in adapting that style?": """Understanding the historical context provides insight into the social, political, and cultural influences that shaped an art style. This knowledge helps artists appreciate the original intentions and themes behind the style, allowing for a more authentic adaptation. It also inspires artists to incorporate contemporary elements, creating a bridge between past and present.
""",
    "What are some exercises to practice transitioning between art styles?": """Execises:
1. Style Study: Recreate a piece of art from a different style, focusing on techniques and principles.
2. Mixed Media: Combine elements from multiple styles in one piece to explore how they interact.
3. Timed Sketches: Create quick sketches in different styles to loosen up and adapt techniques rapidly.
4. Subject Variation: Paint the same subject in various styles (realism, impressionism, abstract) to understand differences.
5. Critique Sessions: Join art groups for feedback on style transitions and to gain new perspectives.
""",

    "How can digital tools aid in adapting different art styles?": """Digital tools offer various advantages, such as:
1. Undo/Redo: Allows for easy correction and experimentation.
2. Brush Variability: Digital platforms provide a wide range of brushes and textures that mimic traditional media.
3. Layering: Artists can work in layers, making it easier to experiment with different styles without altering the original artwork.
4. Filters and Effects: Tools like filters can help artists achieve stylistic effects quickly.
5. Tutorials and Resources: Access to online tutorials and communities can provide guidance and inspiration.
""",
}



# Function to get BERT embeddings for a given text
def get_bert_embedding(text):
    inputs = qa_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = qa_bert_model(**inputs)  # Use qa_bert_model here
    return outputs.last_hidden_state.mean(dim=1)  # Mean pooling to get a single vector

# Precompute embeddings for predefined questions
predefined_questions = list(qa_pairs.keys())
predefined_embeddings = [get_bert_embedding(question) for question in predefined_questions]

# Function to find the best match for the user's question using fuzzywuzzy and BERT
def find_best_match(user_question, fuzzy_threshold=70, similarity_threshold=0.5):
    # First use fuzzywuzzy to find a match
    best_fuzzy_match, fuzzy_score = process.extractOne(user_question, predefined_questions)

    # If fuzzy match is good enough (score above threshold), use it
    if fuzzy_score >= fuzzy_threshold:
        # Check BERT similarity to ensure it’s a relevant match
        user_embedding = get_bert_embedding(user_question)
        best_match_embedding = get_bert_embedding(best_fuzzy_match)
        similarity = cosine_similarity(user_embedding, best_match_embedding)[0][0]
        if similarity >= similarity_threshold:
            return best_fuzzy_match, qa_pairs[best_fuzzy_match]
    
    # Otherwise, provide a fallback answer
    return "Sorry, I couldn't find an answer for that.", None

# Define the Gradio interface functions
def answer_question(user_input):
    matched_question, answer = find_best_match(user_input)
    return answer if answer else "Sorry, I couldn't find an answer for that."

def display_answer(question):
    return qa_pairs.get(question, "Sorry, I couldn't find an answer for that.")

# Create the Gradio interface
def create_interface():
    with gr.Blocks() as demo:
        gr.Markdown("## Canvas Queries")
        gr.Markdown("### Ask, Learn, and Explore the World of Creativity!")

        # Add buttons for each predefined question
        with gr.Row():
            buttons = [gr.Button(question, elem_id=question) for question in predefined_questions]
        
        # Textbox and submit button
        with gr.Row():
            textbox = gr.Textbox(label="Ask a question", placeholder="Type your question here...")
            submit_button = gr.Button("Submit")
            answer_output = gr.Textbox(label="Answer", placeholder="The answer will appear here...")

        # Connect the input box and submit button to the answer function
        submit_button.click(answer_question, inputs=textbox, outputs=answer_output)

        # Connect each button to display the corresponding answer
        for button in buttons:
            button.click(display_answer, inputs=button, outputs=answer_output)

    return demo

iface2 = create_interface()

@trial.route('/art_qa')
def art_qa():
    return render_template('art_qa.html')


# Initialize the art progress database
def init_db():
    conn = sqlite3.connect('progress.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS progress (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            similarity REAL,
            feedback TEXT,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()
init_db()

# Image comparison function with motivational feedback
def compare_images(ref_img, user_img):
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    user_gray = cv2.cvtColor(user_img, cv2.COLOR_BGR2GRAY)

    # Resize user image to match reference
    user_gray = cv2.resize(user_gray, (ref_gray.shape[1], ref_gray.shape[0]))

    # Find differences
    diff = cv2.absdiff(ref_gray, user_gray)
    _, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)

    # Calculate similarity percentage
    similarity = 100 - (cv2.countNonZero(thresh) / thresh.size * 100)
    similarity = round(similarity, 2)  # Round to 2 decimal places

    # Generate motivational feedback based on similarity
    feedback = ""
    if similarity > 90:
        feedback = "Excellent work! Your drawing is very close to the reference. Keep up the great effort!"
    elif similarity > 80:
        feedback = "Fantastic! Your drawing is quite similar to the reference. Just a few more tweaks needed!"
    elif similarity > 70:
        feedback = "Very good! Your drawing shows strong resemblance to the reference. Great job!"
    elif similarity > 60:
        feedback = "Good effort! There are some differences, but you're on the right track."
    elif similarity > 50:
        feedback = "Nice try! You're making progress. Pay more attention to the details for better results."
    elif similarity > 40:
        feedback = "Decent attempt. Focus on refining your drawing to better match the reference."
    elif similarity > 30:
        feedback = "Fair effort. There's room for improvement. Keep practicing and adjust your details."
    elif similarity > 20:
        feedback = "You've made a start. Try to focus more on the reference image for better accuracy."
    elif similarity > 10:
        feedback = "Initial effort is seen. Work on aligning your drawing with the reference for improvement."
    else:
        feedback = "Keep practicing! There's a lot of room for improvement. Review the reference image closely."

    return similarity, feedback

# Route for uploading images and calculating progress
@trial.route('/upload', methods=['POST'])
def upload_images():
    title = request.form['title']
    ref_file = request.files['reference_image']
    user_file = request.files['user_image']

    ref_path = os.path.join(trial.config['UPLOAD_FOLDER'], ref_file.filename)
    user_path = os.path.join(trial.config['UPLOAD_FOLDER'], user_file.filename)

    ref_file.save(ref_path)
    user_file.save(user_path)

    ref_img = cv2.imread(ref_path)
    user_img = cv2.imread(user_path)

    similarity, feedback = compare_images(ref_img, user_img)

    # Save the progress to the database
    conn = sqlite3.connect('progress.db')
    cursor = conn.cursor()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute('''
        INSERT INTO progress (title, similarity, feedback, timestamp)
        VALUES (?, ?, ?, ?)
    ''', (title, similarity, feedback, timestamp))
    conn.commit()
    conn.close()

    os.remove(ref_path)
    os.remove(user_path)

    return redirect(url_for('pro'))

@trial.route('/pro')
def pro():
    return render_template('pro.html')

# Route for displaying progress
@trial.route('/progress')
def display_progress():
    conn = sqlite3.connect('progress.db')
    cursor = conn.cursor()
    cursor.execute('SELECT title, similarity, feedback, timestamp FROM progress')
    rows = cursor.fetchall()
    conn.close()

    progress_data = [{'title': row[0], 'similarity': row[1], 'feedback': row[2], 'timestamp': row[3]} for row in rows]
    return render_template('progress.html', progress_data=progress_data)


@trial.route('/gallery')
def gallery():
    return render_template('gallery.html')

if __name__ == '__main__':
    try:
        threading.Thread(target=lambda: iface2.launch(server_name="127.0.0.1", server_port=7861, share=True)).start()
        threading.Thread(target=lambda: iface1.launch(server_name="127.0.0.1", server_port=8081, share=True)).start()
        
        trial.run(debug=True, port=5000)

    except Exception as e:
        print("Error: ", e)

