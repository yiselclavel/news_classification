package clasificacion_noticias.procesamiento;

import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import java.io.File; 
import java.io.FileReader;
import org.apache.lucene.analysis.StandardAnalyzer;
import org.apache.lucene.document.Field.TermVector;

public class IndexarDocumento {
   
    
      // Directorio donde esta el indice o donde se va a crear
      public String indexFile = "indiceCreado";      

      // Para comprobar si es necesario crear o no el directorio del indice
      public boolean crearDirectorio(){	
	File directorio;
	     if ((directorio = new File(indexFile)).exists() && directorio.isDirectory())
		// Ya existe, no es necesario crearlo
		return false;
		// No existe, es necesario crearlo
		return true;}
        
     // Indexamos el fichero de texto
     public void indexFile(File directorio) throws Exception{
	// Para poder escribir en el indice, se utilizo: IndexWriter	
         IndexWriter writer = new IndexWriter(indexFile, new StandardAnalyzer(), crearDirectorio());                   

	// Vamos a annadir un documento al indice (objeto de la clase Document)

        //Un objeto Document representa un unico documento, 
        //modelado como un conjunto de campos (Fields) de la forma <nombre, valor>
        //Su constructor crea un documento sin ningun campo
	Document doc = new Document();

        //Para añadirles campos, se utiliza el metodo add al que pasamos el campo como parametro.
	        //Constructor: Field(String name, String value, Field.Store store, Field.Index index,TermVector termVector)
                                    //String name: "path" Indica el nombre del campo
                                    //String value: directorio.getPath() El valor a almacenar en el campo
                                    //Store store: Field.Store.NO Si el campo sera indexado para las busquedas
                                    //Index index: Field.Index.UN_TOKENIZED Si el valor de un campo sera "tokenizado" previamente a que sea indexado.
                                    //TermVector termVector: Field.TermVector.YES Si se desea que el valor de un campo sea alamacenado en el indice. En el caso de que el contenido sea razonablemente pequeño, Lucene permite que este sea almacenado en el indice. Con los campos almacenados en el indice, en lugar de utilzar el Docuemnto para localizar el fichero o los datos originales, se pueden recuperar directamente en el indice.
	 //Este campo(Filed) es para almacenar el camino donde esta el fichero
         Field path = new Field("path", directorio.getPath(), Field.Store.NO, Field.Index.UN_TOKENIZED);
         doc.add(path); //Se adiciona ese campo al documento    
	
         //Este campo es para almacenar el contenido del fichero que ya es tokenizado y reducido su dimensionalidad previamente
        Field contenido = new Field("content", new FileReader(directorio),TermVector.WITH_POSITIONS_OFFSETS);
	doc.add(contenido); //El valor original no se almacena en el indice

        // Annadimos el documento al indice
	writer.addDocument(doc);
 	writer.optimize();
        //El mçetodo close()se llama para liberar todos los recursos asociados a la creación del índice.
	writer.close();     
     }
}
