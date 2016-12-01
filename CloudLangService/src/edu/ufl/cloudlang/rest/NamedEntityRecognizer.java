/**
 * This class provides the endpoint for the entity recognition task.
 */
package edu.ufl.cloudlang.rest;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

import javax.ws.rs.GET;
import javax.ws.rs.Path;
import javax.ws.rs.PathParam;
import javax.ws.rs.Produces;
import javax.ws.rs.core.Response;

import org.json.JSONException;
import org.json.JSONObject;

/**
 * @author Sayak Biswas
 *
 */

@Path("/entities")
public class NamedEntityRecognizer {
	@GET
	@Produces("application/json")
	public Response parseText() throws JSONException {
		JSONObject json = new JSONObject();
		json.put("testentitykey", "testentityvalue");
		
		return Response.status(200).entity(json.toString()).build();
	}
	
	@Path("{text}")
	@GET
	@Produces("application/json")
	public Response parseTextWithInput(@PathParam("text") String text) throws JSONException {
		System.out.println("Input Text :: " + text);
		
		String inputPath = "/home/sayak/Workspace/gaurav/input.txt";
		BufferedWriter inputWriter = null;
		String line = null;
		try {
			inputWriter = new BufferedWriter(new FileWriter(new File(inputPath)));
			inputWriter.write(text);
		} catch (IOException ie) {
			ie.printStackTrace();
		} finally {
			if(inputWriter != null) {
				try {
					inputWriter.close();
				} catch (IOException ie2) {
					ie2.printStackTrace();
				}
			}
		}
		
		String outputPath = "/home/sayak/Workspace/gaurav/output.txt";
		String modelPath = "/home/sayak/Workspace/gaurav/models/english/";
		String command = "python /home/sayak/Workspace/gaurav/tagger.py --model " 
							+ modelPath + " --input /home/sayak/Workspace/gaurav/input.txt --output " + outputPath;
		try {
			System.out.println("Will execute command " + command);
			Process process = Runtime.getRuntime().exec(command);
			process.waitFor();
		} catch (IOException ie) {
			ie.printStackTrace();
		} catch (InterruptedException ie) {
			ie.printStackTrace();
		}
		
		JSONObject json =  new JSONObject();
		json.put("inputText", text);
		StringBuilder nerJSON = null;
		BufferedReader inputReader = null;
		line = null;
		try {
			inputReader = new BufferedReader(new FileReader(new File(outputPath)));
			nerJSON = new StringBuilder();
			while((line = inputReader.readLine()) != null) {
				System.out.println(line);
				nerJSON.append(line);
				nerJSON.append("\n");
			}
		} catch (FileNotFoundException re) {
			re.printStackTrace();
		} catch (IOException ie) {
			ie.printStackTrace();
		} finally {
			try {
				if(inputReader != null) {
					inputReader.close();
				}
			} catch (IOException ie) {
				ie.printStackTrace();
			}
		}
		json.put("nerResult", nerJSON.toString());
		
		return Response.status(200).entity(json.toString()).build();
	}
}
