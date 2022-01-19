package irony_detection.util;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.net.URL;
import java.net.URLConnection;
import java.util.StringTokenizer;
import javax.swing.text.BadLocationException;
import javax.swing.text.ChangedCharSetException;
import javax.swing.text.Document;
import javax.swing.text.html.HTMLDocument;
import javax.swing.text.html.HTMLEditorKit;

/**
 *
 * @author admin
 */
public class TextoHTML {
    
    public HTMLDocument loadDocument(URL url, String charSet)
            throws IOException, BadLocationException {
        HTMLDocument doc = new HTMLDocument();
        doc.putProperty(Document.StreamDescriptionProperty, url);

        /*
         * This loop allows the document read to be retried if
         * the character encoding changes during processing.
         */
        InputStream in = null;
        boolean ignoreCharSet = false;

        for (;;) {
            try {
                // Remove any document content
                doc.remove(0, doc.getLength());

                URLConnection urlc = url.openConnection();
                in = urlc.getInputStream();
                Reader reader = (charSet == null) ? new InputStreamReader(in) : new InputStreamReader(in, charSet);

                HTMLEditorKit.Parser p = getParser();
                HTMLEditorKit.ParserCallback htmlReader = getParseCallback(doc);
                p.parse(reader, htmlReader, ignoreCharSet);
                htmlReader.flush();

                // All done
                break;
            } catch (BadLocationException ex) {
                // Should not happen - throw an IOException
                throw new IOException(ex.getMessage());
            } catch (ChangedCharSetException e) {
                // The character set has changed - restart
                charSet = getNewCharSet(e);

                // Prevent recursion by suppressing
                // further exceptions
                ignoreCharSet = true;

                // Close original input stream
                in.close();

                // Continue the loop to read with the correct
                // encoding
            }
        }

        return doc;
    }

    public synchronized HTMLEditorKit.Parser getParser() {
        HTMLEditorKit.Parser parser = null;
        if (parser == null) {
            try {
                Class c = Class.forName("javax.swing.text.html.parser.ParserDelegator");
                parser = (HTMLEditorKit.Parser) c.newInstance();
            } catch (Throwable e) {
            }
        }
        return parser;
    }

    public synchronized HTMLEditorKit.ParserCallback getParseCallback(HTMLDocument doc) {
        return doc.getReader(0);
    }

    protected String getNewCharSet(
            ChangedCharSetException e) {
        String spec = e.getCharSetSpec();
        if (e.keyEqualsCharSet()) {
            // The event contains the new CharSet
            return spec;
        }

        // The event contains the content type
        // plus ";" plus qualifiers which may
        // contain a "charset" directive. First
        // remove the content type.
        int index = spec.indexOf(";");
        if (index != -1) {
            spec = spec.substring(index + 1);
        }

        // Force the string to lower case
        spec = spec.toLowerCase();

        StringTokenizer st =
                new StringTokenizer(spec, " \t=", true);
        boolean foundCharSet = false;
        boolean foundEquals = false;
        while (st.hasMoreTokens()) {
            String token = st.nextToken();
            if (token.equals(" ") || token.equals("\t")) {
                continue;
            }
            if (foundCharSet == false &&
                    foundEquals == false &&
                    token.equals("charset")) {
                foundCharSet = true;
                continue;
            } else if (foundEquals == false &&
                    token.equals("=")) {
                foundEquals = true;
                continue;
            } else if (foundEquals == true &&
                    foundCharSet == true) {
                return token;
            }

            // Not recognized
            foundCharSet = false;
            foundEquals = false;
        }
        //No charset found - return a guess
        return "8859_1";
    }

}
