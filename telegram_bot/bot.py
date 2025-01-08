import logging
import torch
import cv2
import numpy as np
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from PIL import Image
import io
import albumentations as albu
import albumentations.pytorch as albu_pytorch
import time

# Telegram bot token
TELEGRAM_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"

# Device for PyTorch (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained model
# for server /root/bot_tg/deeplabplus_mobile0nes4_epoch10_binary.pth
model = torch.load('background_delete_bot/model_training/deeplabplus_mobile0nes4_epoch10_binary.pth', map_location = device)
model.to(device)
model.eval()

# Logging configuration
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# Augmentation for input images
def get_val_test_augmentation():
    val_test_transforms = [
        albu.PadIfNeeded(512, 512),   # Padding to 512x512 size
        albu.Resize(height=512, width=512, p=1), # Resizing
        albu.Normalize(),   # Normalization
        albu_pytorch.transforms.ToTensorV2()   # Normalization
    ]
    return albu.Compose(val_test_transforms)

# Create a green background
def create_green_background(shape):
    return np.full(shape, [0, 255, 0], dtype=np.uint8)

# Resize or crop the background to match the input image size
def resize_or_crop_background(background, target_shape):
    target_height, target_width = target_shape[:2]
    background_height, background_width = background.shape[:2]

    # Scale the background
    if background_height < target_height or background_width < target_width:
        scale_height = target_height / background_height
        scale_width = target_width / background_width
        scale = max(scale_height, scale_width)

        new_height = int(background_height * scale)
        new_width = int(background_width * scale)
        background = cv2.resize(background, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        background_height, background_width = background.shape[:2]

    # Center and crop the background
    center_y, center_x = background_height // 2, background_width // 2
    crop_y1 = center_y - target_height // 2
    crop_y2 = crop_y1 + target_height
    crop_x1 = center_x - target_width // 2
    crop_x2 = crop_x1 + target_width

    background_resized = background[crop_y1:crop_y2, crop_x1:crop_x2]

    return background_resized

# Process the image: segmentation and background replacement
def process_image(image, background):
    transform = get_val_test_augmentation()

    # Transform the input image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tran_image = transform(image=image)
    
    # Perform segmentation using the model
    mask = model(tran_image['image'].unsqueeze(0).to(device))
    outputs_masks = torch.argmax(mask, 1).squeeze().cpu().numpy()
    
    # Resize the mask to match the original image
    resized_mask = cv2.resize(outputs_masks, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Replace the background
    zero_mask = resized_mask == 0
    background_resized = resize_or_crop_background(background, image.shape)
    result_image = np.where(zero_mask[..., None], background_resized, image).astype(np.uint8)

    return result_image

# Handle the /start command
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        context.user_data.clear()
        await update.message.reply_text("Привет! Отправь мне фотографию, и я уберу с нее фон!\nHi! Send me a photo, and I'll remove the background for you!")
    except Exception as e:
        logger.error(f"Ошибка в команде /start\nError in the /start command: {e}")
        time.sleep(7)
        try:
            await update.message.reply_text("Произошла ошибка, попробуйте снова!\nAn error occurred, please try again!")
        except Exception as inner_e:
            logger.error(f"Ошибка отправки сообщения об ошибке\nError sending the error message: {inner_e}")

# Handle photo messages
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        # If a background is awaited
        if context.user_data.get("awaiting_background"):
            photo = update.message.photo[-1]
            file = await photo.get_file()

            # Load the background image
            img_bytes = await file.download_as_bytearray()
            img_array = np.asarray(bytearray(img_bytes), dtype=np.uint8)
            background = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
            
            # Process the main image
            main_image = context.user_data.pop("main_image")
            context.user_data.pop("awaiting_background", None)
            result_image = process_image(main_image, background)

            # Send the resulting image
            byte_io = io.BytesIO()
            result_image_pil = Image.fromarray(result_image)
            result_image_pil.save(byte_io, 'PNG')
            byte_io.seek(0)
            
            await update.message.reply_photo(photo=byte_io)
            await update.message.reply_text("Готово! Если хотите обработать новое изображение, просто отправьте его снова.\nDone! If you want to process a new image, just send it again.")
        else:
            # Save the main image
            photo = update.message.photo[-1]
            file = await photo.get_file()

            img_bytes = await file.download_as_bytearray()
            img_array = np.asarray(bytearray(img_bytes), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            context.user_data.clear()
            context.user_data["main_image"] = img
            await update.message.reply_text("Изображение получено! Вы хотите задать свой фон?\nImage received! Would you like to set a custom background?")
    except Exception as e:
        logger.error(f"Ошибка в обработке фотографии\nError processing the photo: {e}")
        context.user_data.clear() 
        time.sleep(7)
        try:
            await update.message.reply_text("Произошла ошибка, попробуйте снова!\nAn error occurred, please try again!")
        except Exception as inner_e:
            logger.error(f"Ошибка отправки сообщения об ошибке\nError sending the error message: {inner_e}")

# Handle text messages
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        user_response = update.message.text.strip().lower()
        
        if "main_image" not in context.user_data:
            await update.message.reply_text("Сначала отправьте изображение.\nFirst send the image.")
            return

        # If the user answered "yes"
        if user_response.lower() in ["yes", "yeah", "of course", "да", "конечно"]:
            context.user_data["awaiting_background"] = True
            await update.message.reply_text("Хорошо, теперь отправьте изображение фона.\nOkay, now send the background image.")

        # If the user answered "no"
        elif user_response.lower() in ["no", "not", "нет"]:
            main_image = context.user_data.pop("main_image")
            green_background = create_green_background(main_image.shape[:2] + (3,))
            result_image = process_image(main_image, green_background)

            byte_io = io.BytesIO()
            result_image_pil = Image.fromarray(result_image)
            result_image_pil.save(byte_io, 'PNG')
            byte_io.seek(0)
            
            await update.message.reply_photo(photo=byte_io)
            await update.message.reply_text("Готово! Если хотите обработать новое изображение, отправьте его снова.\nDone! If you want to process a new image, send it again.")
        else:
            await update.message.reply_text("Пожалуйста, ответьте 'да' или 'нет'.\nPlease answer 'yes' or 'no'.")
    except Exception as e:
        logger.error(f"Ошибка в обработке текста\nError in text processing.: {e}")
        context.user_data.clear()
        time.sleep(7)
        try:
            await update.message.reply_text("Произошла ошибка, попробуйте снова!\nAn error occurred, please try again!")
        except Exception as inner_e:
            logger.error(f"Ошибка отправки сообщения об ошибке\nError sending the error message: {inner_e}")

# Main function to run the bot
def main():
    application = Application.builder().token(TELEGRAM_TOKEN).connect_timeout(60).read_timeout(60).build()

    # Command and message handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    
    # Run the bot
    application.run_polling()

if __name__ == "__main__":
    main()
