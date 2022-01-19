/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package irony_detection;

import irony_detection.clasificacion.ClasificarNoticias;
import irony_detection.util.Categoria;
import irony_detection.util.Noticia;
import java.util.LinkedList;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author Yisel
 */
public class Irony_detection {

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
            Logger.getLogger(Irony_detection.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

}
