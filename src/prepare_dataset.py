import os
import shutil
import random
from sklearn.model_selection import train_test_split

def preparar_dataset_yolo():
    """
    Prepara dataset YOLO a partir de carpetas por clase
    """
    
    # Configuración
    clases = {
        'Fito': 0,      # ID 0 para YOLO
        'Monilia': 1,   # ID 1 para YOLO  
        'Sana': 2       # ID 2 para YOLO
    }
    
    # Rutas
    raw_path = 'data/raw'
    processed_path = 'data/processed'
    
    # Crear estructura de salida
    for split in ['train', 'val', 'test']:
        os.makedirs(f'{processed_path}/{split}/images', exist_ok=True)
        os.makedirs(f'{processed_path}/{split}/labels', exist_ok=True)
    
    print("=" * 50)
    print("PREPARADOR DE DATASET - ENFERMEDADES DE CACAO")
    print("=" * 50)
    
    # Verificar carpetas de entrada
    for clase in clases.keys():
        clase_path = os.path.join(raw_path, clase)
        if not os.path.exists(clase_path):
            print(f"⚠️  Advertencia: No existe {clase_path}")
            print(f"   Crea la carpeta y coloca allí las imágenes de {clase}")
            return
    
    # Recolectar todas las imágenes por clase
    todas_imagenes = []
    
    for clase_name, clase_id in clases.items():
        clase_path = os.path.join(raw_path, clase_name)
        imagenes = [f for f in os.listdir(clase_path) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"\n📂 {clase_name}: {len(imagenes)} imágenes")
        
        for img in imagenes:
            todas_imagenes.append({
                'clase': clase_name,
                'clase_id': clase_id,
                'nombre': img,
                'ruta': os.path.join(clase_path, img)
            })
    
    if not todas_imagenes:
        print("\n❌ No se encontraron imágenes")
        return
    
    print(f"\n📊 TOTAL: {len(todas_imagenes)} imágenes")
    
    # Mezclar aleatoriamente
    random.shuffle(todas_imagenes)
    
    # Dividir en train/val/test (70/20/10)
    train_val, test = train_test_split(todas_imagenes, test_size=0.1, random_state=42)
    train, val = train_test_split(train_val, test_size=0.222, random_state=42)  # 20% del total
    
    splits = {
        'train': train,
        'val': val,
        'test': test
    }
    
    # Procesar cada split
    for split_name, split_data in splits.items():
        print(f"\n🔧 Procesando {split_name.upper()} ({len(split_data)} imágenes):")
        
        for i, img_info in enumerate(split_data):
            # Copiar imagen
            img_dest = f"{processed_path}/{split_name}/images/{img_info['clase']}_{img_info['nombre']}"
            shutil.copy2(img_info['ruta'], img_dest)
            
            # Crear archivo de etiqueta YOLO (vacío por ahora)
            # TÚ deberás etiquetar después
            label_name = f"{img_info['clase']}_{os.path.splitext(img_info['nombre'])[0]}.txt"
            label_path = f"{processed_path}/{split_name}/labels/{label_name}"
            
            with open(label_path, 'w') as f:
                # Formato YOLO: clase_id x_center y_center width height
                # Ejemplo: "0 0.5 0.5 0.3 0.3"
                # Por ahora vacío - tú añadirás las anotaciones
                pass
            
            if i < 3:  # Mostrar primeros 3 ejemplos
                print(f"   {img_info['clase']}_{img_info['nombre']} → Clase ID: {img_info['clase_id']}")
    
    # Crear archivo data.yaml
    crear_data_yaml(clases)
    
    print("\n" + "=" * 50)
    print("✅ DATASET PREPARADO CORRECTAMENTE")
    print("=" * 50)
    print("\n📁 Estructura creada en: data/processed/")
    print("\n📝 SIGUIENTES PASOS:")
    print("1. Etiqueta las imágenes usando:")
    print("   • LabelImg (https://github.com/heartexlabs/labelImg)")
    print("   • Roboflow (https://roboflow.com)")
    print("2. Guarda las anotaciones en formato YOLO")
    print("3. Coloca los archivos .txt en las carpetas labels/")
    print("\n🎯 Clases configuradas:")
    for clase, id_clase in clases.items():
        print(f"   • {clase}: ID {id_clase}")

def crear_data_yaml(clases):
    """Crea archivo de configuración para YOLO"""
    
    contenido = f"""# Dataset: Enfermedades del Cacao
# Creado automáticamente

# Rutas (relativas a este archivo)
path: ./data/processed
train: train/images
val: val/images
test: test/images

# Número de clases
nc: {len(clases)}

# Nombres de clases
names:"""
    
    for clase, id_clase in sorted(clases.items(), key=lambda x: x[1]):
        contenido += f"\n  {id_clase}: {clase}"
    
    contenido += "\n\n# Información adicional"
    contenido += "\n# Las imágenes deben estar en formato YOLO"
    contenido += "\n# Anotaciones en archivos .txt con formato:"
    contenido += "\n# clase_id x_center y_center width height"
    
    with open('data/processed/data.yaml', 'w', encoding='utf-8') as f:
        f.write(contenido)
    
    print(f"\n📄 Archivo data.yaml creado en: data/processed/data.yaml")

if __name__ == "__main__":
    preparar_dataset_yolo()