            # # Proceeding with next selection
            # if 'next-selection-button' in triggered_id and n_clicks_next > 0:
            #     logging.info(f"Selecting next artifact window: triggered ID: {triggered_id}")
                
            #     # Reset start and end inputs
            #     start_input = ''
            #     end_input = ''
            #     logging.info(f"Reset start and end inputs: start_input={start_input}, end_input={end_input}")

            #     saved_figure_json = go.Figure(fig)  # Save current figure state
                
            #     # Reset the artifact selection process for next selection
            #     confirm_button_style = {'display': 'block'}
            #     next_button_style = {'display': 'none'}
            #     cancel_button_style = {'display': 'none'}
            #     confirmation_text = f"Previous artifact window confirmed."
                
            #     # Trigger the client-side callback to switch mode back to peak correction
            #     trigger_mode_change = 'Switch to Peak Correction'
                
            #     return [fig, dash.no_update, dash.no_update, dash.no_update, dash.no_update, 
            #             f"Proceeding with next selection...", existing_artifact_windows, start_input, end_input, cancel_button_style, 
            #             trigger_mode_change, dash.no_update, confirm_button_style, confirmation_text,
            #             dash.no_update, dash.no_update, 0, next_button_style]
        # else:
        #     cancel_button_style = {'display': 'none'} if not existing_artifact_windows else {'display': 'block'}
