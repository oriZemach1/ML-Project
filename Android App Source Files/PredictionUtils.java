package com.example.modelapplication;

import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.Matrix;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.imgproc.Imgproc;
import org.opencv.photo.Photo;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

public class PredictionUtils {

    private static final int IMAGE_WIDTH = 128;
    private static final int IMAGE_HEIGHT = 32;
    private static final int TIME_STEPS = 64;
    private static final int CLASS_NUM = 80;
    private final char[] vocab;

    public PredictionUtils() {
        this.vocab = new char[]{' ', '!', '"', '#', '&', '\'', '(', ')', '*', '+', ',', '-', '.',
                '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A',
                'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
                'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'};
    }

    public int getImageWidth() {
        return IMAGE_WIDTH;
    }

    public int getImageHeight() {
        return IMAGE_HEIGHT;
    }

    public int[] getShapeBeforeArgmax() {
        // returns the output's shape after applying the softmax and before calling argmax.
        return new int[] {1, TIME_STEPS, CLASS_NUM};
    }

    public float[][] reshapeResult(float[] flattened) {
        float[][] res = new float[TIME_STEPS][CLASS_NUM];
        for(int i = 0; i < flattened.length; i++) {
            res[i/CLASS_NUM][i%CLASS_NUM] = flattened[i];
        }
        return res;
    }

    public int[] argmax(float[][] result) {
        // preforms argmax on each time-step.

        int[] res = new int[TIME_STEPS];
        for(int i = 0; i < TIME_STEPS; i++) {
            int maxIndex = 0;
            for(int j = 1; j < CLASS_NUM; j++) {
                if(result[i][j] > result[i][maxIndex])
                    maxIndex = j;
            }
            res[i] = maxIndex;
        }
        return res;
    }

    public String ctcDecode(int[] preds) {
        // a greedy decoder for one sample.

        String res = "";

        int prev = -1;
        for(int i = 0; i < preds.length; i++) {
            if(prev == -1 || preds[i] != preds[prev]) {
                prev = i;
                if(preds[i] != vocab.length) // not the empty char
                    res += this.vocab[preds[i]];
            }
        }
        return res;
    }

    public int toGray(int color) {
        return (int) (0.21*Color.red(color) + 0.72*Color.green(color) + 0.07*Color.blue(color));
    }

    public Mat processDocument(Bitmap img) {
        Mat mat = new Mat();
        Utils.bitmapToMat(img, mat);

        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_BGR2GRAY); // convert to grayscale

        Photo.fastNlMeansDenoising(mat, mat, 10); // denoising

        //apply threshold
        Imgproc.adaptiveThreshold(mat, mat, 255,
                Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY, 15, 30);

        return mat;
    }

    public List<Rect> getBoundingRects(Mat img, int iterations) {

        Mat mat = new Mat();
        img.copyTo(mat);

        //dilation
        Core.bitwise_not(mat, mat);
        Mat kernel = Mat.ones(3, 3, CvType.CV_8UC1); // the type is just uint8
        Imgproc.dilate(mat, mat, kernel, new Point(1, 1), iterations);
        Core.bitwise_not(mat, mat);


        List<MatOfPoint> contours = new ArrayList<>();
        List<Rect> rects = new ArrayList<>();
        Mat hierarchy = new Mat();

        Imgproc.findContours(mat, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

        for(MatOfPoint contour : contours) {
            int thresh = 750;
            Rect bound = Imgproc.boundingRect(contour);
            if(bound.area() > thresh && bound.height < mat.rows() && bound.width < mat.cols())
                rects.add(bound);
        }

        return rects;
    }

    public double calcToleranceFromSlider1(float val1) {
        // convert the float scale in range [0, 10] to [0.7, 2.7]

        return 0.7 + (2-val1/5);
    }

    public int calcIterationsFromSlider2(float val2) {
        // convert the float scale in range [0,10] to [10, 25]

        return 10 + (int) (15-val2*1.5);
    }

    public List<List<Rect>> sortRectangles(List<Rect> rectangles, double tolerance) {

        // note: tolerance indicates how much can the user write in non straight lines.

        // Group rectangles into lines
        List<List<Rect>> lines = new ArrayList<>();
        for (Rect rect : rectangles) {
            // Find the line that this rectangle belongs to
            boolean foundLine = false;
            for (List<Rect> line : lines) {
                if (Math.abs(rect.y - line.get(0).y) <= line.get(0).height * tolerance) {
                    line.add(rect);
                    foundLine = true;
                    break;
                }
            }
            if (!foundLine) { // No existing line found, create a new one
                List<Rect> newLine = new ArrayList<>();
                newLine.add(rect);
                lines.add(newLine);
            }
        }

        // Sort the lines by y-coordinate
        lines.sort(Comparator.comparingInt(line -> line.get(0).y));

        // Sort the rectangles within each line by x-coordinate
        for (List<Rect> line : lines)
            line.sort(Comparator.comparingInt(rect -> rect.x));

        return lines;

    }

    public Bitmap subImage(Mat img, Rect rect) {
        Mat res = img.submat(rect);
        Bitmap bitmap = Bitmap.createBitmap(res.cols(), res.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(res, bitmap);

        return bitmap;
    }


}
