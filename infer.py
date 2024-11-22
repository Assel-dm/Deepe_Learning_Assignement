#Accept command-line arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
args = parser.parse_args()

#Load the model and checkpoint
model = build_unet()
model.load_state_dict(torch.load("checkpoints/best_model.pth"))
model.eval()

#Process the input image
image = cv2.imread(args.image_path)
transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2(),
])
input_image = transform(image=image)["image"].unsqueeze(0)

#Predict and save the output
with torch.no_grad():
    output = model(input_image)
segmented_image = (output.squeeze().numpy() > 0.5).astype('uint8')
cv2.imwrite("segmented_image.png", segmented_image * 255)
