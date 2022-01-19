/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package clasificacion_noticias;

import clasificacion_noticias.clasificacion.ClasificarNoticias;
import clasificacion_noticias.util.Categoria;
import clasificacion_noticias.util.Noticia;
import java.util.LinkedList;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author Yisel
 */
public class Clasificacion_noticias {

    /**
     * @param args the command line arguments
     */
    public static LinkedList<Categoria> categorias;
    static LinkedList<Noticia> noticiasExternasClasificadas; //noticias externas clasificadas
    static LinkedList<Noticia> otrasNoticiasClasificadas; //noticias externas originales clasificadas
    private String[] meses = {"Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"};

    public static void main(String[] args) {
        try {
            // TODO code application logic here

            noticiasExternasClasificadas = ClasificarNoticias.classifyNews();
            ClasificarNoticias.train(); //entrenar clasificador

        } catch (Exception ex) {
            Logger.getLogger(Clasificacion_noticias.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

}
