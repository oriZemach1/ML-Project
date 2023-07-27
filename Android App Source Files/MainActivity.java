package com.example.modelapplication;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.app.Activity;
import android.app.AlertDialog;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.example.modelapplication.ml.Model;
import com.google.android.material.slider.Slider;
import com.theartofdev.edmodo.cropper.CropImage;
import com.theartofdev.edmodo.cropper.CropImageView;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    private ImageView imgView;
    private Button camera, predict, select, edit, copy;
    private Slider slider1, slider2;
    private TextView tv;
    private Bitmap img;
    private final int CAMERA_CODE = 100;
    private final int CAMERA_REQ = 101;
    private final int SELECT_CODE = 102;
    private PredictionUtils utils;
    private Uri imageUri;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        getCameraPermission();

        if(OpenCVLoader.initDebug()) Log.i("OpenCV load", "success!");
        else Log.e("OpenCV load", "error!");

        this.imgView = (ImageView) findViewById(R.id.imageView);
        this.camera = (Button) findViewById(R.id.camera);
        this.predict = (Button) findViewById(R.id.predict);
        this.select = (Button) findViewById(R.id.select);
        this.edit = (Button) findViewById(R.id.edit);
        this.copy = (Button) findViewById(R.id.copy);

        this.slider1 = (Slider) findViewById(R.id.slider1);
        this.slider2 = (Slider) findViewById(R.id.slider2);

        this.tv = (TextView) findViewById(R.id.textView);

        this.tv.setMovementMethod(new ScrollingMovementMethod());

        this.utils = new PredictionUtils();

        imageUri = getContentResolver().insert(
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI, new ContentValues());

        copy.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                ClipboardManager clipboard = (ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);
                ClipData clip = ClipData.newPlainText("predicted", tv.getText().toString());
                clipboard.setPrimaryClip(clip);
            }
        });

        camera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                intent.putExtra(MediaStore.EXTRA_OUTPUT, imageUri);
                startActivityForResult(intent, CAMERA_CODE);
            }
        });

        select.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
                intent.setType("image/*");
                startActivityForResult(intent, SELECT_CODE);
            }
        });

        edit.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if(img != null) {
                    CropImage.activity(imageUri).start(MainActivity.this);
                }
            }
        });


        predict.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                // make sure we have an image to predict...
                if(img == null) {
                    invokeAlert("Please load an image before making a prediction.");
                    return;
                }

                // make sure we have the configurations
                if(!checkIfAnswered()) {
                    invokeAlert("Please answer the questions to configure the model");
                    return;
                }

                // Load the TensorFlow Lite model from the asset folder
                Interpreter tflite = null;
                try {
                    tflite = new Interpreter(loadModelFile(MainActivity.this));
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }

                assert tflite != null;

                //predict the text:
                String result = predictDocument(tflite, img);
                tv.setText(result);
            }
        });
    }

    private void invokeAlert(String message) {
        AlertDialog.Builder builder = new AlertDialog.Builder(MainActivity.this);
        builder.setTitle("Alert");
        builder.setMessage(message);
        builder.setPositiveButton("OK", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {}
        });
        AlertDialog dialog = builder.create();
        dialog.show();
    }

    public boolean checkIfAnswered() {
        return this.slider1.getValue() > 0 && this.slider2.getValue() > 0;
    }


    private String predictDocument(Interpreter tflite, Bitmap document) {
        Mat doc = utils.processDocument(document);
        List<Rect> rects = utils.getBoundingRects(doc, utils.calcIterationsFromSlider2(slider2.getValue()));
        List<List<Rect>> lines = utils.sortRectangles(rects, utils.calcToleranceFromSlider1(slider1.getValue()));

        String result = "";
        for(List<Rect> line : lines) {
            for(Rect word : line) {
                Bitmap subImg = utils.subImage(doc, word);
                String cur = predictWord(tflite, subImg);
                result += cur + " ";
            }
            result += "\n";
        }

        return result;
    }


    private String predictWord(Interpreter tflite, Bitmap word) {

        //reshape
        word = Bitmap.createScaledBitmap(word, utils.getImageWidth(),
                utils.getImageHeight(), true);

        // Get the input and output shapes of the model
        int inputIndex = 0;
        int[] inputShape = tflite.getInputTensor(inputIndex).shape();

        ByteBuffer inputBuffer = getInput(tflite, word);

        // Get the output shape and type
        int outputIndex = 0;
        int[] outputShape = utils.getShapeBeforeArgmax();
        DataType outputType = tflite.getOutputTensor(outputIndex).dataType();

        // Initialize the output buffer
        TensorBuffer outputBuffer = TensorBuffer.createFixedSize(outputShape, outputType);

        // Run inference on the input data
        tflite.run(inputBuffer, outputBuffer.getBuffer());

        // Get the output data from the buffer
        float[] outputData = outputBuffer.getFloatArray();

        //decode the output to text
        float[][] res = utils.reshapeResult(outputData);
        int[] encoded = utils.argmax(res);
        String predictedText = utils.ctcDecode(encoded);

        return predictedText;

    }

    private ByteBuffer getInput(Interpreter tflite, Bitmap img) {

        // Get the input shape of the model
        int inputIndex = 0;
        int[] inputShape = tflite.getInputTensor(inputIndex).shape();

        // Create a ByteBuffer to hold the input data
        ByteBuffer inputBuffer = ByteBuffer.allocateDirect(
                inputShape[1] * inputShape[2] * 4); // 4 bytes in every float

        // Set the order of the input buffer to little endian
        inputBuffer.order(ByteOrder.nativeOrder());


        // Get the pixels from the Mat and put them into the input buffer
        Bitmap bitmap = img;
        int[] intValues = new int[inputShape[1] * inputShape[2]];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for (int i = 0; i < inputShape[1]; i++) {
            for (int j = 0; j < inputShape[2]; ++j) {
                 int val = intValues[pixel++];
                inputBuffer.putFloat(utils.toGray(val) / 255.0f);
            }
        }

        return inputBuffer;

    }

    // Helper function to load the model file from the assets folder
    private MappedByteBuffer loadModelFile(Activity activity) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd("model.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if(requestCode == SELECT_CODE && resultCode == RESULT_OK && data != null) {
            try {
                img = MediaStore.Images.Media.getBitmap(this.getContentResolver(), data.getData());
                imgView.setImageBitmap(img);
                imageUri = data.getData();

            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        else if(requestCode == CAMERA_CODE && resultCode == RESULT_OK) {

            try {
                img = MediaStore.Images.Media.getBitmap(getContentResolver(), imageUri);
                imgView.setImageBitmap(img);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        else if (requestCode == CropImage.CROP_IMAGE_ACTIVITY_REQUEST_CODE) {
            CropImage.ActivityResult result = CropImage.getActivityResult(data);
            if (resultCode == RESULT_OK) {
                Uri resultUri = result.getUri();
                try {
                    img = MediaStore.Images.Media.getBitmap(getContentResolver(), resultUri);
                    imgView.setImageBitmap(img);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            } else if (resultCode == CropImage.CROP_IMAGE_ACTIVITY_RESULT_ERROR_CODE) {
                Exception error = result.getError();
            }
        }
    }

    private void getCameraPermission() {
        if(checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED)
            requestPermissions(new String[] {Manifest.permission.CAMERA}, CAMERA_REQ);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if(requestCode == CAMERA_REQ && grantResults.length > 0) {
            if(grantResults[0] != PackageManager.PERMISSION_GRANTED)
                getCameraPermission();
        }
    }
}