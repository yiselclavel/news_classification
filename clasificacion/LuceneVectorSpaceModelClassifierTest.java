package irony_detection.clasificacion;

/*import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.Map;

import org.apache.commons.collections15.Transformer;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang.StringUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.commons.math.linear.RealMatrix;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.Field.Index;
import org.apache.lucene.document.Field.Store;
import org.apache.lucene.document.Field.TermVector;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriter.MaxFieldLength;
import org.apache.lucene.store.FSDirectory;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

import com.mycompany.myapp.indexers.IdfIndexer;
import com.mycompany.myapp.indexers.TfIndexer;
import com.mycompany.myapp.similarity.CosineSimilarity;
import com.mycompany.myapp.summarizers.SummaryAnalyzer;

public class LuceneVectorSpaceModelClassifierTest {
private static final Log log = LogFactory.getLog(LuceneVectorSpaceModelClassifierTest.class);

  private static String INPUT_FILE =
    "src/test/resources/data/sugar-coffee-cocoa-docs.txt";
  private static String INDEX_DIR = "src/test/resources/data/scc-index";
  private static String[] DOCS_TO_CLASSIFY = new String[] {
    "src/test/resources/data/cocoa.txt",
    "src/test/resources/data/cocoa1.txt",
    "src/test/resources/data/cocoa2.txt",
    "src/test/resources/data/coffee.txt",
    "src/test/resources/data/coffee1.txt"
  };

  @BeforeClass
  public static void buildIndex() throws Exception {
    log.debug("Building index...");
    BufferedReader reader = new BufferedReader(new FileReader(INPUT_FILE));
    IndexWriter writer = new IndexWriter(FSDirectory.getDirectory(INDEX_DIR),new SummaryAnalyzer(), MaxFieldLength.UNLIMITED);
    String line = null;
    int lno = 0;
    StringBuilder bodybuf = new StringBuilder();
    String category = null;
    while ((line = reader.readLine()) != null) {
      if (line.endsWith(".sgm")) {
        // header line
        if (lno > 0) {
          // not the very first line, so dump current body buffer and
          // reinit the buffer.
          writeToIndex(writer, category, bodybuf.toString());
          bodybuf = new StringBuilder();
        }
        category = StringUtils.trim(StringUtils.split(line, ":")[1]);
        continue;
      } else {
        // not a header line, accumulate line into bodybuf
        bodybuf.append(line).append(" ");
      }
      lno++;
    }
    // last record
    writeToIndex(writer, category, bodybuf.toString());
    reader.close();
    writer.commit();
    writer.optimize();
    writer.close();
  }

  private static void writeToIndex(IndexWriter writer, String category, String body) throws Exception {
    Document doc = new Document();
    doc.add(new Field("category", category, Store.YES, Index.NOT_ANALYZED));
    doc.add(new Field("body", body, Store.YES, Index.ANALYZED, TermVector.YES));
    writer.addDocument(doc);
  }

  //@AfterClass
  public static void deleteIndex() throws Exception {
    log.info("Deleting index directory...");
    FileUtils.deleteDirectory(new File(INDEX_DIR));
  }

  //@Test
  public void testLuceneNaiveBayesClassifier() throws Exception {
    LuceneVectorSpaceModelClassifier classifier = new LuceneVectorSpaceModelClassifier();
    // setup
    classifier.setIndexDir(INDEX_DIR);
    classifier.setAnalyzer(new SummaryAnalyzer());
    classifier.setCategoryFieldName("category");
    classifier.setBodyFieldName("body");
    // this is the default but we set it anyway, to illustrate usage
    classifier.setIndexers(new Transformer[] {
      new TfIndexer(),
      new IdfIndexer()
    });
    // this is the default but we set it anyway, to illustrate usage.
    // Similarity need not be set before training, it can be set before
    // the classification step.
    classifier.setSimilarity(new CosineSimilarity());
    // training
    classifier.train();
    // classification
    Map<String,RealMatrix> centroidMap = classifier.getCentroidMap();
    Map<String,Integer> termIdMap = classifier.getTermIdMap();
    String[] categories = centroidMap.keySet().toArray(new String[0]);
    for (String testDoc : DOCS_TO_CLASSIFY) {
      File f = new File(testDoc);
      String category = classifier.classify(centroidMap, termIdMap,
        FileUtils.readFileToString(f, "UTF-8"));
      System.out.println(">>> " + f.getName() +
        " => category: " + category);
      Map<String,Double> similarityMap = classifier.getSimilarityMap();
      String[] pairs = new String[categories.length];
      for (int i = 0; i < categories.length; i++) {
         pairs[i] = categories[i] + ":" + similarityMap.get(categories[i]);
      }
      System.out.println("(" + StringUtils.join(pairs, ", ") + ")");
    }
  }

}*/
