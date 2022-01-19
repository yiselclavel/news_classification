package clasificacion_noticias.util;

import java.io.File;
import java.util.Date;
import java.util.LinkedList;

/**
 *
 * @author Yisel
 */
public class Noticia {

    private int id;
    private File noticia;
    private String titulo;
    private String texto; //texto original
    private Categoria categoria; //categoria almacenada en la BD
    private LinkedList<String> categoriasPropuestas; //categorias propuestas por el clasificador
    private int posicion;
    private String autor;
    private Date fecha;
    private String fuente;

////noticias clasificadas
//    public Noticia(int p, File n, LinkedList<String> cat) {
//        posicion = p;
//        noticia = n;
//        categoriasPropuestas = cat;
//        fecha = null;
//        fuente = "Ahora";
//        autor = null;
//        texto = null;
//        categoria = null;
//    }
//noticias de entrenamiento
    public Noticia(File f, String ti, String te, Categoria c) {
        id = -1;
        noticia = f;
        posicion = -1;
        titulo = ti;
        texto = te;
        categoria = c;
        categoriasPropuestas = null;
        fecha = null;
        fuente = "Ahora";
        autor = null;
    }

//noticias internas
    public Noticia(int id,File f, String ti, String t, Categoria c, LinkedList<String> cat, Date d, String a) {
        this.id = id;
        noticia = f;
        posicion = -1;
        titulo = ti;
        texto = t;
        categoria = c;
        categoriasPropuestas = cat;
        fecha = d;
        fuente = "Ahora";
        autor = a;
    }

//noticias externas
    public Noticia(int p, File f, String ti, String t, long d) {
        id = -1;
        noticia = f;
        posicion = p;
        titulo = ti;
        texto = t;
        categoriasPropuestas = null;
        fecha = new Date(d);
        fuente = null;
        categoria = null;
        autor = null;
    }

    /**
     * @return the noticia
     */
    public File getNoticia() {
        return noticia;
    }

    /**
     * @param noticia the noticia to set
     */
    public void setNoticia(File noticia) {
        this.noticia = noticia;
    }

    /**
     * @return the categorias
     */
    public LinkedList<String> getCategoriasPropuestas() {
        return categoriasPropuestas;
    }

    /**
     * @param categorias the categorias to set
     */
    public void setCategoriasPropuestas(LinkedList<String> categorias) {
        this.categoriasPropuestas = categorias;
    }

    /**
     * @return the pos
     */
    public int getPosicion() {
        return posicion;
    }

    /**
     * @param pos the pos to set
     */
    public void setPosicion(int pos) {
        this.posicion = pos;
    }

    /**
     * @return the fecha
     */
    public Date getFecha() {
        return fecha;
    }

    /**
     * @param fecha the fecha to set
     */
    public void setFecha(Date fecha) {
        this.fecha = fecha;
    }

    /**
     * @return the fuente
     */
    public String getFuente() {
        return fuente;
    }

    /**
     * @param fuente the fuente to set
     */
    public void setFuente(String fuente) {
        this.fuente = fuente;
    }

//    public String toString() {
//        return noticia.getName() + " " + categorias.toString() + " " + getFecha().toString() + " " + getFuente();
//    }
    public String toString() {
        return titulo /*+ " " + categoriasPropuestas.toString()*/;
    }

    /**
     * @return the categoriaAutor
     */
    public Categoria getCategoria() {
        return categoria;
    }

    /**
     * @param categoriaAutor the categoriaAutor to set
     */
    public void setCategoria(Categoria categoria) {
        this.categoria = categoria;
    }

    /**
     * @return the autor
     */
    public String getAutor() {
        return autor;
    }

    /**
     * @param autor the autor to set
     */
    public void setAutor(String autor) {
        this.autor = autor;
    }

    /**
     * @return the texto
     */
    public String getTexto() {
        return texto;
    }

    /**
     * @param texto the texto to set
     */
    public void setTexto(String texto) {
        this.texto = texto;
    }

    /**
     * @return the titulo
     */
    public String getTitulo() {
        return titulo;
    }

    /**
     * @param titulo the titulo to set
     */
    public void setTitulo(String titulo) {
        this.titulo = titulo;
    }

    /**
     * @return the id
     */
    public int getId() {
        return id;
    }

    /**
     * @param id the id to set
     */
    public void setId(int id) {
        this.id = id;
    }
    /**
     * @return the texto1
     */
}
