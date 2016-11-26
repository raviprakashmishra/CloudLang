/**
 * This class provides the endpoint for the entity recognition task.
 */
package edu.ufl.cloudlang.rest;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
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
		
		String result = "@Produces(\"application/json\") Output: \n\n Parse result: \n\n" + json;
		return Response.status(200).entity(result).build();
	}
	
	@Path("{text}")
	@GET
	@Produces("application/json")
	public Response parseTextWithInput(@PathParam("text") String text) throws JSONException {
		System.out.println("Input Text :: " + text);
		
		String inputPath = "/home/sayak/Workspace/gaurav/input.txt";
		FileOutputStream fileOutputStream = null;
		File file = null;
		try {
			file = new File(inputPath);
			fileOutputStream = new FileOutputStream(file);
			if(!file.exists()) {
				file.createNewFile();
			}
			byte[] inputInBytes = text.getBytes();
			fileOutputStream.write(inputInBytes);
			fileOutputStream.flush();
		} catch (IOException ie) {
			ie.printStackTrace();
		} finally {
			if(fileOutputStream != null) {
				try {
					fileOutputStream.close();
				} catch (IOException ie2) {
					ie2.printStackTrace();
				}
			}
		}
		
		String outputPath = "/home/sayak/Workspace/gaurav/output.txt";
		String modelPath = "/home/sayak/Workspace/gaurav/models/english/";
		String command = "python /home/sayak/Workspace/gaurav/tagger.py --model " 
							+ modelPath + " --input input.txt --output " + outputPath;
		try {
			System.out.println("Will execute command " + command);
			Runtime.getRuntime().exec(command);
		} catch (IOException ie) {
			ie.printStackTrace();
		}
		
		JSONObject json =  new JSONObject();
		json.put("input text", text);
		StringBuilder nerJSON = null;
		BufferedReader inputReader = null;
		String line = null;
		try {
			inputReader = new BufferedReader(new FileReader(new File(outputPath)));
			nerJSON = new StringBuilder();
			while((line = inputReader.readLine()) != null) {
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
		json.put("nerResult", nerJSON);
		
		String result = "@Produces(\"application/json\") Output: \n\n Parse result: \n\n" + json;
		return Response.status(200).entity(result).build();
	}
}
