package irony_detection.util;

import com.aliasi.util.Files;
import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;

/**
 *
 * @author Yisel
 */
public class FilesUtil {

//    public static long readPosInConfigurationFile(File f, String text) throws IOException {
//        RandomAccessFile out = new RandomAccessFile(f, "r");
//        long pos = out.getFilePointer();
//        String line = null;
//        while (!(line = out.readLine()).equals(null)) {
//            if (line.split("=")[0].equalsIgnoreCase(text)) {
//                return pos;
//            }
//            pos = out.getFilePointer();
//        }
//        return -1;
//    }
    
    public static String readFromConfigurationFile(String option) {
        String text = null;
        try {
            text = Files.readFromFile(new File("datos/Configuración.conf"), "ISO-8859-1");
        } catch (IOException io) {
            io.printStackTrace();
        }
        String[] lines = text.split("\n");
        String[] fields = null;
        for (int i = 0; i < lines.length; ++i) {
            fields = lines[i].split("=");
            if (fields[0].equalsIgnoreCase(option)) {
                return fields[1];
            }
        }
        return "";
    }

    public static void writeToConfigurationFile(String[] text) throws IOException {
        RandomAccessFile out = new RandomAccessFile(new File("datos/Configuración.conf"), "rw");
        out.setLength(0);
        for (int i = 0; i < text.length; i++) {
            out.writeBytes(text[i] + "\n");
        }
        out.close();
    }

    
}
