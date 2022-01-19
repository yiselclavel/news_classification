package clasificacion_noticias.clasificacion;

/*import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import net.sf.jtmt.clustering.ByValueComparator;
import net.sf.jtmt.indexers.IdfIndexer;
import net.sf.jtmt.indexers.TfIndexer;
import net.sf.jtmt.similarity.AbstractSimilarity;
import net.sf.jtmt.similarity.CosineSimilarity;

import org.apache.commons.collections15.Bag;
import org.apache.commons.collections15.Transformer;
import org.apache.commons.collections15.bag.HashBag;
import org.apache.commons.collections15.comparators.ReverseComparator;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.commons.math.linear.Array2DRowRealMatrix;
import org.apache.commons.math.linear.OpenMapRealMatrix;
import org.apache.commons.math.linear.RealMatrix;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.StandardAnalyzer;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.Field.Index;
import org.apache.lucene.document.Field.Store;
import org.apache.lucene.document.Field.TermVector;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.TermEnum;
import org.apache.lucene.index.TermFreqVector;
import org.apache.lucene.index.IndexWriter.MaxFieldLength;
import org.apache.lucene.store.RAMDirectory;

public class LuceneVectorSpaceModelClassifier {
 private final Log log = LogFactory.getLog(getClass());

  private String indexDir;
  private String categoryFieldName;
  private String bodyFieldName;

  private Analyzer analyzer = new StandardAnalyzer();

  @SuppressWarnings("unchecked")
  private Transformer<RealMatrix,RealMatrix>[] indexers = new Transformer[] {
    new TfIndexer(),
    new IdfIndexer()
  };

  private AbstractSimilarity similarity = new CosineSimilarity();

  private Map<String,RealMatrix> centroidMap;
  private Map<String,Integer> termIdMap;
  private Map<String,Double> similarityMap;*/

  /**
   * Set the directory where the Lucene index is located.
   * @param indexDir the index directory.
   */
 /* public void setIndexDir(String indexDir) {
    this.indexDir = indexDir;
  }
*/
  /**
   * Set the name of the Lucene field containing the preclassified category.
   * @param categoryFieldName the category field name.
   */
/*  public void setCategoryFieldName(String categoryFieldName) {
    this.categoryFieldName = categoryFieldName;
  }*/

  /**
   * Set the name of the Lucene field containing the document body. The
   * document body must have been indexed with TermVector.YES.
   * @param bodyFieldName the name of the document body field.
   */
 /* public void setBodyFieldName(String bodyFieldName) {
    this.bodyFieldName = bodyFieldName;
  }
*/
  /**
   * The Analyzer used for tokenizing the document body during indexing,
   * and to tokenize the text to be classified. If not specified, the
   * classifier uses the StandardAnalyzer.
   * @param analyzer the analyzer reference.
   */
 /* public void setAnalyzer(Analyzer analyzer) {
    this.analyzer = analyzer;
  }*/

  /**
   * A transformer chain of indexers (or normalizers) to normalize the
   * document matrices. If not specified, the default is a chain of TF/IDF.
   * @param indexers the normalizer chain.
   */
 /* public void setIndexers(Transformer<RealMatrix,RealMatrix>[] indexers) {
    this.indexers = indexers;
  }*/

  /**
   * The Similarity implementation used to calculate the similarity between
   * the text to be classified and the category centroid document matrices.
   * Uses CosineSimilarity if not specified.
   * @param similarity the similarity to use.
   */
 /* public void setSimilarity(AbstractSimilarity similarity) {
    this.similarity = similarity;
  }*/

  /**
   * Implements the logic for training the classifier. The input is a Lucene
   * index of preclassified documents. The classifier is provided the name
   * of the field which contains the document body, as well as the name of
   * the field which contains the category name. Additionally, the document
   * body must have had its Term Vectors computed during indexing (using
   * TermVector.YES). The train method uses the Term Vectors to compute a
   * geometrical centroid for each category in the index. The set of category
   * names to their associated centroid document matrix is available via the
   * getCentroidMap() method after training is complete.
   * @throws Exception if one is thrown.
   */
  /*public void train() throws Exception {
    train(null);
  }*/

  /**
   * Slightly specialized version of the classifier train() method that
   * takes a Set of docIds. This is for doing cross-validation. If a null
   * Set of docIds is passed in, then it uses the entire training set.
   * @param docIds a Set of docIds to consider for training. If null, the
   *               all the docIds are considered for training.
   * @throws Exception if thrown.
   */
 /* public void train(Set<Integer> docIds) throws Exception {
    log.info("Classifier training started");
    IndexReader reader = IndexReader.open(indexDir);
    // Set up a data structure for the term versus the row id in the matrix.
    // This is going to be used for looking up the term's row in the matrix.
    this.termIdMap = computeTermIdMap(reader);
    // Initialize the data structures to hold the td matrices for the various
    // categories.
    Bag<String> docsInCategory = computeDocsInCategory(reader);
    Map<String,Integer> currentDocInCategory = new HashMap<String,Integer>();
    Map<String,RealMatrix> categoryTfMap = new HashMap<String,RealMatrix>();
    for (String category : docsInCategory.uniqueSet()) {
      int numDocsInCategory = docsInCategory.getCount(category);
      categoryTfMap.put(category,
        new OpenMapRealMatrix(termIdMap.size(), numDocsInCategory));
      currentDocInCategory.put(category, new Integer(0));
    }*/
    // extract each document body's TermVector into the td matrix for
    // that document's category
   /* int numDocs = reader.numDocs();
    for (int i = 0; i < numDocs; i++) {
      Document doc = reader.document(i);
      if (docIds != null && docIds.size() > 0) {
        // check to see if the current document is in our training set,
        // and if so, train with it
        if (! docIds.contains(i)) {
          continue;
        }
      }
      String category = doc.get(categoryFieldName);
      RealMatrix tfMatrix = categoryTfMap.get(category);
      // get the term frequency map
      TermFreqVector vector = reader.getTermFreqVector(i, bodyFieldName);
      String[] terms = vector.getTerms();
      int[] frequencies = vector.getTermFrequencies();
      for (int j = 0; j < terms.length; j++) {
        int row = termIdMap.get(terms[j]);
        int col = currentDocInCategory.get(category);
        tfMatrix.setEntry(row, col, new Double(frequencies[j]));
      }
      incrementCurrentDoc(currentDocInCategory, category);
    }
    reader.close();
    // compute centroid vectors for each category
    this.centroidMap = new HashMap<String,RealMatrix>();
    for (String category : docsInCategory.uniqueSet()) {
      RealMatrix tdmatrix = categoryTfMap.get(category);
      RealMatrix centroid = computeCentroid(tdmatrix);
      centroidMap.put(category, centroid);
    }
    log.info("Classifier training complete");
  }*/

  /**
   * Returns the centroid map of category name to TD Matrix containing the
   * centroid document of the category. This data is computed as a side
   * effect of the train() method.
   * @return the centroid map computed from the training.
   */
 /* public Map<String,RealMatrix> getCentroidMap() {
    return centroidMap;
  }*/

  /**
   * Returns the map of analyzed terms versus their positions in the centroid
   * matrices. The data is computed as a side-effect of the train() method.
   * @return a Map of analyzed terms to their position in the matrix.
   */
 /* public Map<String,Integer> getTermIdMap() {
    return termIdMap;
  }*/

  /**
   * Once the classifier is trained using the train() method, it creates a
   * Map of category to associated centroid documents for each category, and
   * a termIdMap, which is a mapping of tokenized terms to its row number in
   * the document matrix for the centroid documents. These two structures are
   * used by the classify method to match up terms from the input text with
   * corresponding terms in the centroids to calculate similarities. Builds
   * a Map of category names and the similarities of the input text to the
   * centroids in each category as a side effect. Returns the category with
   * the highest similarity score, ie the category this text should be
   * classified under.
   * @param centroids a Map of category names to centroid document matrices.
   * @param termIdMap a Map of terms to their positions in the document matrix.
   * @param text the text to classify.
   * @return the best category for the text.
   * @throws Exception if one is thrown.
   */
 /* public String classify(Map<String,RealMatrix> centroids,
      Map<String,Integer> termIdMap, String text) throws Exception {
    RAMDirectory ramdir = new RAMDirectory();
    indexDocument(ramdir, "text", text);
    // now find the (normalized) term frequency vector for this
    RealMatrix docMatrix = buildMatrixFromIndex(ramdir, "text");
    // compute similarity using passed in Similarity implementation, we
    // use CosineSimilarity by default.
    this.similarityMap = new HashMap<String,Double>();
    for (String category : centroids.keySet()) {
      RealMatrix centroidMatrix = centroids.get(category);
      double sim = similarity.computeSimilarity(docMatrix, centroidMatrix);
      similarityMap.put(category, sim);
    }
    // sort the categories
    List<String> categories = new ArrayList<String>();
    categories.addAll(centroids.keySet());
    Collections.sort(categories,
      new ReverseComparator<String>(
      new ByValueComparator<String,Double>(similarityMap)));
    // return the best category, the similarity map is also available
    // to the client for debugging or display.
    return categories.get(0);
  }*/

  /**
   * Returns the map of category to similarity with the document after
   * classification. The similarityMap is computed as a side-effect of
   * the classify() method, so the data is interesting only if this method
   * is called after classify() completes successfully.
   * @return map of category to similarity scores for text to classify.
   */
 /* public Map<String,Double> getSimilarityMap() {
    return similarityMap;
  }*/

  /**
   * Loops through the IndexReader's TermEnum enumeration, and creates a Map
   * of term to an integer id. This map is going to be used to assign string
   * terms to specific rows in the Term Document Matrix for each category.
   * @param reader a reference to the IndexReader.
   * @return a Map of terms to their integer ids (0-based).
   * @throws Exception if one is thrown.
   */
  /*private Map<String, Integer> computeTermIdMap(IndexReader reader) throws Exception {
    Map<String,Integer> termIdMap = new HashMap<String,Integer>();
    int id = 0;
    TermEnum termEnum = reader.terms();
    while (termEnum.next()) {
      String term = termEnum.term().text();
      if (termIdMap.containsKey(term)) {
        continue;
      }
      termIdMap.put(term, id);
      id++;
    }
    return termIdMap;
  }*/

  /**
   * Loops through the specified IndexReader and returns a Bag of categories
   * and their document counts. We don't use the BitSet/DocIdSet approach
   * here because we don't know how many categories the training documents have
   * been classified into.
   * @param reader the reference to the IndexReader.
   * @return a Bag of category names and counts.
   * @throws Exception if one is thrown.
   */
 /* private Bag<String> computeDocsInCategory(IndexReader reader) throws Exception {
    int numDocs = reader.numDocs();
    Bag<String> docsInCategory = new HashBag<String>();
    for (int i = 0; i < numDocs; i++) {
      Document doc = reader.document(i);
      String category = doc.get(categoryFieldName);
      docsInCategory.add(category);
    }
    return docsInCategory;
  }*/

  /**
   * Increments the counter for the category to point to the next document
   * index. This is used to manage the document index in the td matrix for
   * the category.
   * @param currDocs the Map of category-wise document Id counters.
   * @param category the category whose document-id we want to increment.
   */
 /* private void incrementCurrentDoc(Map<String,Integer> currDocs, String category) {
    int currentDoc = currDocs.get(category);
    currDocs.put(category, currentDoc + 1);
  }*/

  /**
   * Compute the centroid document from the TD Matrix. Result is a matrix
   * of term weights but for a single document only.
   * @param tdmatrix
   * @return
   */
 /* private RealMatrix computeCentroid(RealMatrix tdmatrix) {
    tdmatrix = normalizeWithTfIdf(tdmatrix);
    RealMatrix centroid = new OpenMapRealMatrix(tdmatrix.getRowDimension(), 1);
    int numDocs = tdmatrix.getColumnDimension();
    int numTerms = tdmatrix.getRowDimension();
    for (int row = 0; row < numTerms; row++) {
      double rowSum = 0.0D;
      for (int col = 0; col < numDocs; col++) {
        rowSum += tdmatrix.getEntry(row, col);
      }
      centroid.setEntry(row, 0, rowSum / ((double) numDocs));
    }
    return centroid;
  }*/

  /**
   * Builds an in-memory Lucene index using the text supplied for classification.
   * @param ramdir the RAM Directory reference.
   * @param fieldName the field name to index the text as.
   * @param text the text to index.
   * @throws Exception if one is thrown.
   */
 /* private void indexDocument(RAMDirectory ramdir, String fieldName, String text)
      throws Exception {
    IndexWriter writer = new IndexWriter(ramdir, analyzer, MaxFieldLength.UNLIMITED);
    Document doc = new Document();
    doc.add(new Field(fieldName, text, Store.YES, Index.ANALYZED, TermVector.YES));
    writer.addDocument(doc);
    writer.commit();
    writer.close();
  }*/

  /**
   * Given a Lucene index and a field name with pre-computed TermVectors,
   * creates a document matrix of terms. The document matrix is normalized
   * using the specified indexer chain.
   * @param ramdir the RAM Directory reference.
   * @param fieldName the name of the field to build the matrix from.
   * @return a normalized Document matrix of terms and frequencies.
   * @throws Exception if one is thrown.
   */
 /* private RealMatrix buildMatrixFromIndex(RAMDirectory ramdir, String fieldName)
      throws Exception {
    IndexReader reader = IndexReader.open(ramdir);
    TermFreqVector vector = reader.getTermFreqVector(0, fieldName);
    String[] terms = vector.getTerms();
    int[] frequencies = vector.getTermFrequencies();
    RealMatrix docMatrix = new OpenMapRealMatrix(termIdMap.size(), 1);
    for (int i = 0; i < terms.length; i++) {
      String term = terms[i];
      if (termIdMap.containsKey(term)) {
        int row = termIdMap.get(term);
        docMatrix.setEntry(row, 0, frequencies[i]);
      }
    }
    reader.close();
    // normalize the docMatrix using TF*IDF
    docMatrix = normalizeWithTfIdf(docMatrix);
    return docMatrix;
  }*/

  /**
   * Pass the input TD Matrix through a chain of transformers to normalize
   * the TD Matrix. Here we do TF/IDF normalization, although it is possible
   * to do other types of normalization (such as LSI) by passing in the
   * appropriate chain of normalizers (or indexers).
   * @param docMatrix the un-normalized TD Matrix.
   * @return the normalized TD Matrix.
   */
  /*private RealMatrix normalizeWithTfIdf(RealMatrix docMatrix) {
    for (Transformer<RealMatrix,RealMatrix> indexer : indexers) {
      docMatrix = indexer.transform(docMatrix);
    }
    return docMatrix;
  }

  public double crossValidate(int folds, int times) throws Exception {
    IndexReader reader = IndexReader.open(indexDir);
    int numDocs = reader.maxDoc();
    int numDocsPerFold = numDocs / folds;
    Set<String> categories = computeDocsInCategory(reader).uniqueSet();
    Map<String,Integer> categoryPosMap = new HashMap<String,Integer>();
    int pos = 0;
    for (String categoryName : categories) {
      categoryPosMap.put(categoryName, pos);
      pos++;
    }
    int numCats = categories.size();
    RealMatrix confusionMatrix =
      new Array2DRowRealMatrix(numCats, numCats);
    for (int i = 0; i < times; i++) {
      for (int j = 0; j < folds; j++) {
        reset();
        Map<String,Set<Integer>> partition = new HashMap<String,Set<Integer>>();
        partition.put("test", new HashSet<Integer>());
        partition.put("train", new HashSet<Integer>());
        Set<Integer> testDocs = generateRandomTestDocs(numDocs, numDocsPerFold);
        for (int k = 0; k < numDocs; k++) {
          if (testDocs.contains(k)) {
            partition.get("test").add(k);
          } else {
            partition.get("train").add(k);
          }
        }
        train(partition.get("train"));
        for (int docId : partition.get("test")) {
          Document testDoc = reader.document(docId);
          String actualCategory = testDoc.get(categoryFieldName);
          String body = testDoc.get(bodyFieldName);
          Map<String,RealMatrix> centroidMap = getCentroidMap();
          Map<String,Integer> termIdMap = getTermIdMap();
          String predictedCategory = classify(centroidMap, termIdMap, body);
          // increment the counter for the confusion matrix
          int row = categoryPosMap.get(actualCategory);
          int col = categoryPosMap.get(predictedCategory);
          confusionMatrix.setEntry(row, col,
            confusionMatrix.getEntry(row, col) + 1);
        }
      }
    }*/
    // print confusion matrix
    /*prettyPrint(confusionMatrix, categoryPosMap);
    // compute accuracy
    double trace = confusionMatrix.getTrace(); // sum of diagnonal elements
    double sum = 0.0D;
    for (int i = 0; i < confusionMatrix.getRowDimension(); i++) {
      for (int j = 0; j < confusionMatrix.getColumnDimension(); j++) {
        sum += confusionMatrix.getEntry(i, j);
      }
    }
    double accuracy = trace / sum;
    return accuracy;
  }

  private Set<Integer> generateRandomTestDocs(int numDocs, int numDocsPerFold) {
    Set<Integer> docs = new HashSet<Integer>();
    while (docs.size() < numDocsPerFold) {
      docs.add((int) (numDocs * Math.random() - 1));
    }
    return docs;
  }

  private void prettyPrint(RealMatrix confusionMatrix,
      Map<String,Integer> categoryPosMap) {
    System.out.println("==== Confusion Matrix ====");
    // invert the map and write the header
    System.out.printf("%10s", " ");
    Map<Integer,String> posCategoryMap = new HashMap<Integer,String>();
    for (String category : categoryPosMap.keySet()) {
      posCategoryMap.put(categoryPosMap.get(category), category);
      System.out.printf("%8s", category);
    }
    System.out.printf("%n");
    for (int i = 0; i < confusionMatrix.getRowDimension(); i++) {
      System.out.printf("%10s", posCategoryMap.get(i));
      for (int j = 0; j < confusionMatrix.getColumnDimension(); j++) {
        System.out.printf("%8d", (int) confusionMatrix.getEntry(i, j));
      }
      System.out.printf("%n");
    }
  }*/

  /**
   * Reset internal data structures. Used by the cross validation process
   * to reset the internal state of the classifier between runs.
   */
 /* private void reset() {
    if (centroidMap != null) {
      centroidMap.clear();
    }
    if (termIdMap != null) {
      termIdMap.clear();
    }
    if (similarityMap != null) {
      similarityMap.clear();
    }
  }

}*/
