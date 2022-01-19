package clasificacion_noticias.util;

/**
 *
 * @author Yisel
 */
public class Categoria {
    private String categoria;
    private String seccion;
   
    public Categoria(String c, String s){
        categoria = c;
        seccion = s;
    }

    /**
     * @return the categoria
     */
    public String getCategoria() {
        return categoria;
    }

    /**
     * @param categoria the categoria to set
     */
    public void setCategoria(String categoria) {
        this.categoria = categoria;
    }

    /**
     * @return the seccion
     */
    public String getSeccion() {
        return seccion;
    }

    /**
     * @param seccion the seccion to set
     */
    public void setSeccion(String seccion) {
        this.seccion = seccion;
    }

    @Override
    public String toString() {
        return "Categoria{" + "categoria=" + categoria + '}';
    }
    
    


}
